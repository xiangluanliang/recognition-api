# 进行id匹配
import numpy as np

def match_person_id(current_centers, prev_centers, threshold=60):
    ids = []
    matched = set()
    for c in current_centers:
        found = False
        for pid, pc in prev_centers.items():
            if pid in matched:
                continue
            dist = np.linalg.norm(np.array(c) - np.array(pc))
            if dist < threshold:
                ids.append(pid)
                matched.add(pid)
                prev_centers[pid] = c
                found = True
                break
        if not found:
            new_id = max(prev_centers.keys(), default=0) + 1
            ids.append(new_id)
            prev_centers[new_id] = c
            matched.add(new_id)
    return ids
