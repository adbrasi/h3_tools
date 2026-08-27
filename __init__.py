"""h3_tools — MiniMax H3 Reference to Video (Pro) + Modal API client nodes."""

import logging

from comfy_api.latest import ComfyExtension

WEB_DIRECTORY = "./web"

try:
    import comfy_extras.nodes_minimax_h3  # noqa: F401
    from .h3_tools.node import H3RefToVideoContinuePro, H3RefToVideoPro
    _NATIVE_NODES = [H3RefToVideoPro, H3RefToVideoContinuePro]
except ImportError:
    _NATIVE_NODES = []
    logging.warning(
        "h3_tools: this ComfyUI has no MiniMax H3 support "
        "(comfy_extras.nodes_minimax_h3) — loading only the Modal API "
        "client nodes. Update ComfyUI to master for the local Pro nodes.")

from .h3_tools.api_nodes import API_NODES  # noqa: E402
from .h3_tools.show_text import SHOW_TEXT_NODES  # noqa: E402


class H3ToolsExtension(ComfyExtension):
    async def get_node_list(self):
        return _NATIVE_NODES + API_NODES + SHOW_TEXT_NODES


async def comfy_entrypoint() -> H3ToolsExtension:
    return H3ToolsExtension()
