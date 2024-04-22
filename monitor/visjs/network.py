# TODO
# from __future__ import annotations

# from collections import defaultdict
# from typing import Union

# from ..html import Component, Options


# class VisNetwork(Component):

#     def __init__(self, name: str):
#         super().__init__(f'vis-network-{name}')
#         self.ids: dict[str, int] = {}
#         self.nodes: list[dict[str, Options]] = []
#         self.edges: list[dict[str, Options]] = []
#         self.options = Options()

#     def add_node(self, id: str, **kwargs):
#         if id not in self.ids:
#             idx = len(self.ids)
#             self.ids[id] = idx
#             self.nodes.append({'id': idx, **kwargs})
#         else:
#             raise ValueError(f'Node "{id}" already exists')

#     def add_edge(self, from_id: str, to_id: str, **kwargs):
#         if from_id in self.ids and to_id in self.ids:
#             kwargs.pop('from', None)
#             kwargs.pop('to', None)
#             self.edges.append(
#                 {'from': self.ids[from_id], 'to': self.ids[to_id], **kwargs})
#         else:
#             raise ValueError(f'Node "{from_id}" or "{to_id}" does not exist')

#     def set_options(self, **kwargs):
#         self.options.merge(kwargs)

#     def reset(self):
#         self.ids.clear()
#         self.nodes.clear()
#         self.edges.clear()
