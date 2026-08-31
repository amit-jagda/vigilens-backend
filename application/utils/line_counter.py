import numpy as np
from collections import defaultdict

class LineCrossingCounter:
    def __init__(self, start_point: list[int] | np.ndarray, end_point: list[int] | np.ndarray, counting_region: int = 15):
        """
        Mathematical line crossing logic based on directional vector geometry.
        """
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.counting_region = counting_region
        
        self.line_vector = self.end_point - self.start_point
        self.line_length = np.linalg.norm(self.line_vector)
        self.unit_line_vector = self.line_vector / self.line_length if self.line_length > 0 else np.array([0, 0])
        self.normal_vector = np.array([-self.unit_line_vector[1], self.unit_line_vector[0]])
        
        dx = abs(self.end_point[0] - self.start_point[0])
        dy = abs(self.end_point[1] - self.start_point[1])
        self.is_vertical = dx < dy
        
        self.object_tracks = defaultdict(list)
        self.armed_side = {}
        self.in_count = 0
        self.out_count = 0
        
    def get_distance_from_line(self, point: np.ndarray) -> float:
        return np.dot(point - self.start_point, self.normal_vector)
        
    def is_projection_on_segment(self, point: np.ndarray) -> bool:
        point_vector = point - self.start_point
        projection_length = np.dot(point_vector, self.unit_line_vector)
        buffer = 50
        return -buffer <= projection_length <= self.line_length + buffer
        
    def update(self, object_id: int, center_point: tuple[float, float]) -> str | bool:
        self.object_tracks[object_id].append(center_point)
        
        current_pos = np.array(self.object_tracks[object_id][-1])
        current_distance = self.get_distance_from_line(current_pos)
        current_side = 1 if current_distance >= 0 else -1
        
        if object_id not in self.armed_side:
            if abs(current_distance) > self.counting_region:
                self.armed_side[object_id] = current_side
            else:
                self.armed_side[object_id] = None
                
        if self.armed_side[object_id] is None:
            if abs(current_distance) > self.counting_region:
                self.armed_side[object_id] = current_side
                
        if self.armed_side[object_id] is None:
            return False
            
        if len(self.object_tracks[object_id]) < 2:
            return False
            
        if current_side != self.armed_side[object_id]:
            prev_side = self.armed_side[object_id]
            self.armed_side[object_id] = None
            
            if self.is_projection_on_segment(current_pos):
                if prev_side == -1 and current_side == 1:
                    self.in_count += 1
                    return "in"
                elif prev_side == 1 and current_side == -1:
                    self.out_count += 1
                    return "out"
        return False
