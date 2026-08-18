"""h3_tools — MiniMax H3 Reference to Video (Pro)."""

try:
    import comfy_extras.nodes_minimax_h3  # noqa: F401
except ImportError as e:
    raise ImportError(
        "h3_tools requires a ComfyUI version with MiniMax H3 support "
        "(comfy_extras.nodes_minimax_h3). Update ComfyUI to the latest master."
    ) from e

from comfy_api.latest import ComfyExtension

from .h3_tools.node import H3RefToVideoPro

WEB_DIRECTORY = "./web"


class H3ToolsExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3RefToVideoPro]


async def comfy_entrypoint() -> H3ToolsExtension:
    return H3ToolsExtension()
