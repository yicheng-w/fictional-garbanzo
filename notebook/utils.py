import csv
from queue import Queue

def find_decision_boundary(predictions, dims, steps):
    N = len(predictions)
    visited = [False for _ in range(N)]
    
    boundaries = []
    
    def get_neighbors(x):
        #print(x)
        
        xit = x
        coords = []
        for d in range(dims):
            coords.append(xit % steps)
            xit = xit // steps
            
        neighbors = []
            
        for d in range(dims):
            multiplier = steps ** d
            
            if coords[d] != 0: # can go lower
                neighbors.append(x - multiplier)
            if coords[d] != (steps - 1): # can go higher
                neighbors.append(x + multiplier)
                
        #print(coords)
        #print(neighbors)
        
        return neighbors
        
    frontiner = Queue()
    
    current = 0
    visited[current] = True
    frontiner.put(current)
           
    while not frontiner.empty():
        current = frontiner.get()
        visited[current] = True
        for n in get_neighbors(current):
            if not visited[n]:
                if predictions[current] != predictions[n]:
                    boundaries.append((current, n))
                visited[n] = True
                frontiner.put(n)
                
    return boundaries

def write_out_csv(data, predictions, boundary, pred2label, out_file_loc):
    with open(out_file_loc, 'w') as out_file:
        fieldnames = ['sent1', 'label1', 'sent2', 'label2']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for s1id, s2id in boundary:
            row = {}
            row['sent1'] = data[s1id]
            row['label1'] = pred2label[predictions[s1id]]
            row['sent2'] = data[s2id]
            row['label2'] = pred2label[predictions[s2id]]
            writer.writerow(row)
    return