"""The ``agentplane`` CLI (SPEC §4.2).

Exit codes: 0 ok, 1 validation failed, 2 transport/auth error, 3 not found.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import TypeAdapter, ValidationError

from agentplane_core import (
    FlowDefinition,
    Resource,
    ValidationResult,
    validate_structure,
)
from agentplane_sdk.client import RegistryClient, RuntimeClient
from agentplane_sdk.config import resolve_config, token_provider_from_config
from agentplane_sdk.errors import (
    AuthError,
    ConflictError,
    NotFoundError,
    TransportError,
    ValidationFailedError,
)

app = typer.Typer(name="agentplane", help="agentplane platform CLI", no_args_is_help=True)
resources_app = typer.Typer(help="Manage runtime resources", no_args_is_help=True)
app.add_typer(resources_app, name="resources")

EXIT_OK = 0
EXIT_VALIDATION = 1
EXIT_TRANSPORT = 2
EXIT_NOT_FOUND = 3

_RESOURCE_ADAPTER: TypeAdapter[Resource] = TypeAdapter(Resource)

RuntimeUrlOption = Annotated[
    str | None, typer.Option("--runtime-url", envvar="AGENTPLANE_RUNTIME_URL")
]
RegistryUrlOption = Annotated[
    str | None, typer.Option("--registry-url", envvar="AGENTPLANE_REGISTRY_URL")
]
TokenOption = Annotated[str | None, typer.Option("--token", envvar="AGENTPLANE_TOKEN")]
JsonFlag = Annotated[bool, typer.Option("--json", help="machine-readable output")]


def _fail(message: str, code: int) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def _run[R](coro: Coroutine[Any, Any, R]) -> R:
    try:
        return asyncio.run(coro)
    except ValidationFailedError as exc:
        _print_issues(exc.result, as_json=False)
        raise typer.Exit(EXIT_VALIDATION) from None
    except NotFoundError as exc:
        _fail(f"not found: {exc}", EXIT_NOT_FOUND)
    except (TransportError, AuthError) as exc:
        _fail(str(exc), EXIT_TRANSPORT)
    except ConflictError as exc:
        _fail(f"conflict: {exc}", EXIT_TRANSPORT)
    raise AssertionError("unreachable")


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        _fail(f"file not found: {path}", EXIT_NOT_FOUND)
    except yaml.YAMLError as exc:
        _fail(f"invalid YAML in {path}: {exc}", EXIT_VALIDATION)
    if not isinstance(data, dict):
        _fail(f"{path} does not contain a mapping", EXIT_VALIDATION)
    return {str(key): value for key, value in data.items()}


def _print_issues(result: ValidationResult, *, as_json: bool) -> None:
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    for issue in result.issues:
        typer.echo(f"{issue.severity.upper()} {issue.code} {issue.path}: {issue.message}", err=True)


def _runtime_client(runtime_url: str | None, token: str | None) -> RuntimeClient:
    config = resolve_config(runtime_url=runtime_url, token=token)
    if not config.runtime_url:
        _fail(
            "no runtime URL configured (flag --runtime-url / env AGENTPLANE_RUNTIME_URL)",
            EXIT_TRANSPORT,
        )
    assert config.runtime_url is not None
    return RuntimeClient(config.runtime_url, token_provider_from_config(config))


def _registry_client(registry_url: str | None, token: str | None) -> RegistryClient:
    config = resolve_config(registry_url=registry_url, token=token)
    if not config.registry_url:
        _fail(
            "no registry URL configured (flag --registry-url / env AGENTPLANE_REGISTRY_URL)",
            EXIT_TRANSPORT,
        )
    assert config.registry_url is not None
    return RegistryClient(config.registry_url, token_provider_from_config(config))


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(help="flow definition YAML")],
    json_output: JsonFlag = False,
    remote: Annotated[
        bool, typer.Option("--remote", help="ask the runtime (authoritative, adds E020-E022)")
    ] = False,
    runtime_url: RuntimeUrlOption = None,
    token: TokenOption = None,
) -> None:
    """Validate a definition; local by default, --remote for the runtime's answer."""
    raw = _load_yaml(file)
    if remote:

        async def remote_validate() -> ValidationResult:
            async with _runtime_client(runtime_url, token) as client:
                return await client.validate(raw)

        result = _run(remote_validate())
    else:
        result = ValidationResult.from_issues(validate_structure(raw))
    _print_issues(result, as_json=json_output)
    raise typer.Exit(EXIT_OK if result.valid else EXIT_VALIDATION)


@app.command()
def deploy(
    file: Annotated[Path, typer.Argument(help="flow definition YAML")],
    draft: Annotated[bool, typer.Option("--draft", help="create/update the draft only")] = False,
    version_label: Annotated[
        str | None,
        typer.Option("--version-label", help="semantic version for this deploy, e.g. 1.2.0"),
    ] = None,
    runtime_url: RuntimeUrlOption = None,
    token: TokenOption = None,
) -> None:
    """Create-or-update the definition by name; deploys unless --draft."""
    raw = _load_yaml(file)
    try:
        defn = FlowDefinition.model_validate(raw)
    except ValidationError:
        result = ValidationResult.from_issues(validate_structure(raw))
        _print_issues(result, as_json=False)
        raise typer.Exit(EXIT_VALIDATION) from None

    async def do_deploy() -> str:
        async with _runtime_client(runtime_url, token) as client:
            try:
                await client.create_draft(defn)
            except ConflictError:
                await client.update_draft(defn.name, defn)
            if draft:
                return f"draft saved: {defn.name}"
            info = await client.deploy(defn.name, version_label=version_label)
            label = f" ({info.version_label})" if info.version_label else ""
            return f"deployed {info.name} v{info.version}{label} -> {info.endpoint_url}"

    typer.echo(_run(do_deploy()))


@app.command()
def undeploy(
    name: str,
    runtime_url: RuntimeUrlOption = None,
    token: TokenOption = None,
) -> None:
    """Stop serving a deployed definition and deregister it."""

    async def do_undeploy() -> None:
        async with _runtime_client(runtime_url, token) as client:
            await client.undeploy(name)

    _run(do_undeploy())
    typer.echo(f"undeployed {name}")


@app.command(name="list")
def list_definitions(
    status: Annotated[str | None, typer.Option("--status")] = None,
    json_output: JsonFlag = False,
    runtime_url: RuntimeUrlOption = None,
    token: TokenOption = None,
) -> None:
    """List definitions known to the runtime."""

    async def do_list() -> list[dict[str, object]]:
        async with _runtime_client(runtime_url, token) as client:
            return [i.model_dump(mode="json") for i in await client.list(status)]

    infos = _run(do_list())
    if json_output:
        typer.echo(json.dumps(infos, indent=2))
        return
    for info in infos:
        endpoint = info.get("endpoint_url") or "-"
        typer.echo(f"{info['name']:<30} {info['status']:<10} {endpoint}")


@app.command()
def export(
    name: str,
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
    version: Annotated[int | None, typer.Option("--version")] = None,
    runtime_url: RuntimeUrlOption = None,
    token: TokenOption = None,
) -> None:
    """Export the canonical serialized definition."""

    async def do_export() -> FlowDefinition:
        async with _runtime_client(runtime_url, token) as client:
            return await client.export(name, version)

    defn = _run(do_export())
    text = yaml.safe_dump(defn.canonical_dict(), sort_keys=False, allow_unicode=True)
    if output is None:
        typer.echo(text)
    else:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {output}")


@app.command()
def search(
    query: Annotated[str, typer.Argument()] = "",
    tags: Annotated[list[str] | None, typer.Option("--tags")] = None,
    semantic: Annotated[bool, typer.Option("--semantic")] = False,
    kind: Annotated[str | None, typer.Option("--kind")] = None,
    json_output: JsonFlag = False,
    registry_url: RegistryUrlOption = None,
    token: TokenOption = None,
) -> None:
    """Search the registry."""

    async def do_search() -> list[dict[str, object]]:
        async with _registry_client(registry_url, token) as client:
            page = await client.search(query, tags=tags, kind=kind, semantic=semantic)
            return [e.model_dump(mode="json") for e in page.items]

    entries = _run(do_search())
    if json_output:
        typer.echo(json.dumps(entries, indent=2))
        return
    for entry in entries:
        card = entry.get("card")
        name = card.get("name", "?") if isinstance(card, dict) else "?"
        typer.echo(f"{name:<30} {entry['kind']:<11} {entry['status']:<9} {entry['url']}")


@resources_app.command(name="list")
def resources_list(
    kind: Annotated[str | None, typer.Option("--kind")] = None,
    json_output: JsonFlag = False,
    runtime_url: RuntimeUrlOption = None,
    token: TokenOption = None,
) -> None:
    """List resources."""

    async def do_list() -> list[dict[str, object]]:
        async with _runtime_client(runtime_url, token) as client:
            resources = await client.list_resources(kind)
        return [_RESOURCE_ADAPTER.dump_python(r, mode="json") for r in resources]

    resources = _run(do_list())
    if json_output:
        typer.echo(json.dumps(resources, indent=2))
        return
    for resource in resources:
        typer.echo(f"{resource['name']:<30} {resource['kind']}")


@resources_app.command(name="create")
def resources_create(
    file: Annotated[Path, typer.Option("-f", "--file", help="resource YAML")],
    runtime_url: RuntimeUrlOption = None,
    token: TokenOption = None,
) -> None:
    """Create a resource from a YAML file."""
    raw = _load_yaml(file)
    try:
        resource = _RESOURCE_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        _fail(f"invalid resource: {exc}", EXIT_VALIDATION)
        raise AssertionError("unreachable") from None

    async def do_create() -> str:
        async with _runtime_client(runtime_url, token) as client:
            created = await client.create_resource(resource)
            return created.name

    typer.echo(f"created resource {_run(do_create())}")


@resources_app.command(name="delete")
def resources_delete(
    name: str,
    runtime_url: RuntimeUrlOption = None,
    token: TokenOption = None,
) -> None:
    """Delete a resource (refused while referenced)."""

    async def do_delete() -> None:
        async with _runtime_client(runtime_url, token) as client:
            await client.delete_resource(name)

    _run(do_delete())
    typer.echo(f"deleted resource {name}")


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
