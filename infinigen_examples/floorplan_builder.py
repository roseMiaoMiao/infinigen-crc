# Copyright (C) 2024
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

import logging

# from dataclasses import dataclass
from typing import Tuple

# import bpy
import numpy as np
import shapely.geometry as sg
from shapely.geometry import Polygon

# from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.example_solver.geometry import parse_scene
from infinigen.core.constraints.example_solver.geometry.planes import Planes
from infinigen.core.constraints.example_solver.room.base import RoomGraph
from infinigen.core.constraints.example_solver.room.solidifier import (
    BlueprintSolidifier,
    Opening,
)
from infinigen.core.constraints.example_solver.state_def import (
    ObjectState,
    RelationState,
    State,
)

# from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


class FloorPlanStateBuilder:
    """Builder class for creating State objects from manual floor plan inputs"""

    def __init__(self, factory_seed=None, consgraph=None):
        self.state = State()
        self.room_contours = {}
        self.openings = []
        self.factory_seed = factory_seed
        self.graphs = []  # Store room graphs
        self.consgraph = consgraph  # Store the constraint graph

        # Constants for solidifier
        self.constants = {
            "wall_thickness": 0.2,
            "wall_height": 3.0,
            "door_width": 1.0,
            "window_width": 1.2,
            "segment_margin": 0.1,
        }

    def add_room(self, name: str, contour: Polygon, room_type: t.Semantics):
        """Add a room with its contour and type"""
        self.room_contours[name] = {"contour": contour, "type": room_type}
        return self

    def add_opening(self, opening: Opening):
        """Add a door or window"""
        self.openings.append(opening)
        return self

    def _point_in_polygon(self, point: Tuple[float, float], polygon: Polygon) -> bool:
        """Helper to check if point is inside polygon"""
        return polygon.intersects(sg.Point(point))

    def _create_room_states(self):
        """Create room states with proper geometry and relationships"""
        # First handle interior room relationships
        for room_name, room_data in self.room_contours.items():
            neighbors = []
            # Find shared edges with other rooms
            for other_name, other_data in self.room_contours.items():
                if other_name != room_name:
                    # Get intersection between room boundaries
                    shared_edge = room_data["contour"].intersection(
                        other_data["contour"]
                    )
                    if not shared_edge.is_empty:
                        # Convert to MultiLineString if necessary
                        if shared_edge.geom_type == "LineString":
                            shared_edge = sg.MultiLineString([shared_edge])
                        elif shared_edge.geom_type == "GeometryCollection":
                            lines = [
                                g
                                for g in shared_edge.geoms
                                if g.geom_type == "LineString"
                            ]
                            if lines:
                                shared_edge = sg.MultiLineString(lines)
                            else:
                                continue
                        elif shared_edge.geom_type != "MultiLineString":
                            continue

                        # Determine connector type based on openings
                        connector_type = (
                            cl.ConnectorType.Door
                            if any(
                                o.type == "door"
                                and room_name in o.connected_rooms
                                and other_name in o.connected_rooms
                                for o in self.openings
                            )
                            else cl.ConnectorType.Wall
                        )

                        # Create relation with shared edge geometry
                        neighbors.append(
                            RelationState(
                                relation=cl.RoomNeighbour({connector_type}),
                                target_name=other_name,
                                value=shared_edge,
                            )
                        )

            # Create room state
            self.state[room_name] = ObjectState(
                obj=None,
                polygon=room_data["contour"],
                tags={t.Semantics.Room, room_data["type"]},
                relations=neighbors,
            )

        # Handle exterior room and its relationships
        exterior_polygon = self._create_exterior_polygon()
        exterior_relations = []

        for room_name, room_data in self.room_contours.items():
            # Get intersection with exterior boundary
            shared_edge = exterior_polygon.intersection(room_data["contour"].boundary)
            if not shared_edge.is_empty:
                # Convert to MultiLineString if necessary
                if shared_edge.geom_type == "LineString":
                    shared_edge = sg.MultiLineString([shared_edge])
                elif shared_edge.geom_type == "GeometryCollection":
                    lines = [
                        g for g in shared_edge.geoms if g.geom_type == "LineString"
                    ]
                    if lines:
                        shared_edge = sg.MultiLineString(lines)
                    else:
                        continue
                elif shared_edge.geom_type != "MultiLineString":
                    continue

                # Create exterior wall relationship
                exterior_relations.append(
                    RelationState(
                        relation=cl.RoomNeighbour({cl.ConnectorType.Wall}),
                        target_name=room_name,
                        value=shared_edge,
                    )
                )

        # Create exterior room state
        self.state["exterior"] = ObjectState(
            obj=None,
            polygon=exterior_polygon,
            tags={t.Semantics.Exterior},
            relations=exterior_relations,
        )

        # Create and store room graph
        self.graphs.append(self._create_room_graph())

    def _create_exterior_polygon(self):
        """Create exterior polygon that contains all rooms with margin"""
        all_coords = []
        for room_data in self.room_contours.values():
            all_coords.extend(room_data["contour"].exterior.coords[:-1])

        bounds = np.array(all_coords)
        min_x, min_y = bounds.min(axis=0) - 1  # 1m margin
        max_x, max_y = bounds.max(axis=0) + 1

        return sg.box(min_x, min_y, max_x, max_y)

    def solve(self):
        """Build state using solidifier"""
        # Create basic state if not already created
        if not self.state.objs:
            self._create_room_states()

        # Create solidifier
        solidifier = BlueprintSolidifier(
            consgraph=self.consgraph,
            graph=self.graphs[0],  # First floor graph
            level=0,  # Ground floor
        )

        # Solidify the state
        new_state, rooms = solidifier.solidify(self.state, self.openings)
        self.state = new_state

        # Calculate dimensions
        bounds = np.array(
            [
                [c["contour"].bounds[0], c["contour"].bounds[1]]
                for c in self.room_contours.values()
            ]
        )
        dimensions = (
            np.max(bounds[:, 0]) - np.min(bounds[:, 0]),
            np.max(bounds[:, 1]) - np.min(bounds[:, 1]),
            self.constants["wall_height"],
        )

        # Add required properties
        self.state.graphs = self.graphs
        self.state.trimesh_scene = parse_scene.parse_scene(
            [o.obj for o in self.state.objs.values() if o.obj is not None]
        )
        self.state.planes = Planes()
        self.state.constants = self.constants

        return self.state, None, dimensions

    def create_example_state(self):
        """Create an example state with two rooms"""

        # Create room contours
        living_room = sg.box(0, 0, 4, 5)
        # bedroom = sg.box(4, 0, 7, 4)
        bedroom = sg.box(4, 0, 6, 3)

        # Add rooms
        self.add_room("living-room_0/0", living_room, t.Semantics.LivingRoom)
        self.add_room("bedroom_0/0", bedroom, t.Semantics.Bedroom)

        # Add doors
        """self.add_opening(Opening(
            #position=(3.5, 2, 4.5, 3),
            position=(4, 1.3, 4.5, 2.3),
            type='door',
            connected_rooms=['bedroom_0/0', 'living-room_0/0'],
            name='door'
        ))"""
        self.add_opening(
            Opening(
                position=(1.5, 4.5, 2.5, 5.5),
                type="door",
                connected_rooms=["living-room_0/0"],
                name="entrance",
            )
        )

        # Add windows
        window_positions = [
            # Bedroom windows - on right wall
            # (5.5, 0.5, 6.5, 1.5),  # Lower window
            # (6.5, 2.5, 7.5, 3.5),  # Upper window
            # Living room windows - on left wall
            (-0.5, 1.0, 0.5, 2.0),  # Lower window
            (-0.5, 3.0, 0.5, 4.0),  # Upper window
        ]

        for pos in window_positions:
            center_point = ((pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2)
            connected_room = (
                ["bedroom_0/0"]
                if self._point_in_polygon(center_point, bedroom)
                else ["living-room_0/0"]
            )

            self.add_opening(
                Opening(position=pos, type="window", connected_rooms=connected_room)
            )

    def _create_room_graph(self):
        """Create RoomGraph according to base.py implementation"""
        # Create adjacency list
        room_indices = {name: i for i, name in enumerate(self.room_contours.keys())}
        children = [[] for _ in range(len(room_indices))]

        # Find neighbors for each room
        for room_name, room_data in self.room_contours.items():
            room_idx = room_indices[room_name]
            for other_name, other_data in self.room_contours.items():
                if other_name != room_name:
                    if room_data["contour"].touches(other_data["contour"]):
                        other_idx = room_indices[other_name]
                        children[room_idx].append(other_idx)

        # Find entrance (if any)
        entrance = None
        for i, (name, _) in enumerate(self.room_contours.items()):
            for opening in self.openings:
                if (
                    opening.type == "door"
                    and opening.name == "entrance"
                    and name in opening.connected_rooms
                ):
                    entrance = i
                    break
            if entrance is not None:
                break

        # Create RoomGraph
        room_graph = RoomGraph(
            children=children, names=list(self.room_contours.keys()), entrance=entrance
        )

        return room_graph
