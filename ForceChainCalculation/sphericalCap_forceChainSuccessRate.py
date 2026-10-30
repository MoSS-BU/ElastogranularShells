import csv
import math
import time
import os
import numpy as np
from scipy.spatial import KDTree, ConvexHull
from collections import deque
import matplotlib.pyplot as plt

# spreads grains evenly on spherical cap using approximately hexagonal packing
def hex_pack_for_N(R, h, d, N):
    cos_tmax = (R - h)/R
    tmax     = math.acos(cos_tmax)
    cap_arc  = R * tmax
    def count_for_s(s):
        dr = s*math.sqrt(3)/2
        kmax = int(cap_arc//dr)
        total = 1
        for k in range(1, kmax+1):
            arc_r = dr*k
            theta = arc_r/R
            rho   = R*math.sin(theta)
            n_k   = int(math.floor(2*math.pi*rho / s))
            total += n_k
        return total

    lo, hi = d, 2*cap_arc
    for _ in range(30):
        mid = 0.5*(lo+hi)
        if count_for_s(mid) >= N:
            lo = mid
        else:
            hi = mid
    s  = lo
    dr = s*math.sqrt(3)/2

    pts   = [(0.0,0.0,R)]
    theta = [0.0]
    tags  = [0]
    k = 1
    while True:
        arc_r = dr*k
        if arc_r > cap_arc:
            break
        th   = arc_r/R
        rho  = R*math.sin(th)
        z    = R*math.cos(th)
        n_k  = int(math.floor(2*math.pi*rho / s))
        if n_k < 1:
            break
        off  = (k%2)*(math.pi/n_k)
        for j in range(n_k):
            phi = off + 2*math.pi*j/n_k
            x   = rho*math.cos(phi); y = rho*math.sin(phi)
            pts.append((x,y,z))
            theta.append(th)
            tags.append(k)
        k += 1

    if len(pts) < N:
        raise RuntimeError(f"Generated only {len(pts)} < requested N={N}")

    data = list(zip(tags, theta, pts))
    data.sort(key=lambda x: x[0])
    data = data[:N]
    tags, theta, pts = zip(*data)
    return np.array(pts), np.array(theta), np.array(tags), s

# perturbs each hexagonally packed point within a wiggle radius to add randomness
def jitter_hex_pts(pts, s, d, R):
    w = max(0.0, (s - d)/2.0)
    new_pts = []
    for C in pts:
        Nrm = C/np.linalg.norm(C)
        ref = np.array([0,0,1.0]) if abs(Nrm[2])<0.9 else np.array([0,1.0,0])
        e1  = np.cross(Nrm, ref); e1 /= np.linalg.norm(e1)
        e2  = np.cross(Nrm, e1)
        r   = w * math.sqrt(np.random.rand())
        a   = 2*math.pi*np.random.rand()
        disp= r*(math.cos(a)*e1 + math.sin(a)*e2)
        C2  = C + disp
        C2  = R*(C2/np.linalg.norm(C2))
        new_pts.append(C2)
    new_theta = [math.acos(p[2]/R) for p in new_pts]
    return np.array(new_pts), np.array(new_theta)

# creates list of all point pairs within set touch distance
def build_graph_fast(pts, R, touch):
    chord = 2*R*math.sin((touch/R)/2)
    tree  = KDTree(pts)
    pairs = tree.query_pairs(chord)
    adj   = [[] for _ in pts]
    for i,j in pairs:
        adj[i].append(j)
        adj[j].append(i)
    return adj

# finds grains at edge of the spherical cap
def find_edge_sets(pts, theta, R, h, d, tags):
    # rim by tags or geometry
    kmax = tags.max()
    by_tag = set(np.where(tags==kmax)[0])
    tmax = math.acos((R-h)/R)
    by_geo = set(np.where(R*(tmax-theta) <= d/2.0)[0])
    rim = by_tag.union(by_geo)
    # hull
    hull = ConvexHull(pts[:,:2])
    hull_idx = set(hull.vertices)
    # union
    edge_all = rim.union(hull_idx)
    return edge_all

# finds path of upward chains for force chain propagation
    # (path from sources to ANY disk in targets set)
def find_chain_upward(adj, pts, sources, targets, max_deg=90.0):
    cos_thresh = math.cos(math.radians(max_deg))
    visited = set(sources)
    q = deque((s, [s]) for s in sources)
    while q:
        u, path = q.popleft()
        if u in targets:
            return path
        Pu = pts[u]
        for v in adj[u]:
            if v in visited:
                continue
            vec = pts[v] - Pu
            dz  = vec[2]
            normv = np.linalg.norm(vec)
            if normv == 0 or dz / normv < cos_thresh:
                continue
            visited.add(v)
            q.append((v, path + [v]))
    return None

#finds all source disks at rim that can be reached
def count_chain_sources(adj, pts, sources, targets, max_deg=90.0):
    """Count how many source disks can reach ANY disk in targets set"""
    cos_thresh = math.cos(math.radians(max_deg))
    count = 0
    
    for start in sources:
        visited = {start}
        q = deque([start])
        found = False
        
        while q and not found:
            u = q.popleft()
            if u in targets:
                found = True
                count += 1
                break
            
            Pu = pts[u]
            for v in adj[u]:
                if v in visited: continue
                vec = pts[v] - Pu
                dz = vec[2]
                if np.linalg.norm(vec) > 0 and dz/np.linalg.norm(vec) >= cos_thresh:
                    visited.add(v)
                    q.append(v)
    
    return count

def main():
    # shell parameters
    cap_diam = 63.75; h = 24.5; d = 2.6
    R       = cap_diam/2

    # user inputs of the program
    max_N = int(math.floor(0.861 * 2*math.pi*R*h / (math.pi*(d/2)**2)))
    print(f"Allowed phi: 0.01-0.861")
    # starting disk value
    N0  = (math.floor(float(input(" Start phi: ")) * (max_N/.861)))
    # ending disk value   
    N1  = (math.floor(float(input(" End phi: ")) * (max_N/.861)))         
    # step size for disk 
    step = 1 #int(input(" Step size: "))           
    
    # calculations
    # number of trials done for each grain value, change if desired
    M   = 100 #int(input("Trials per N: "))
    # max separation in terms of diameter to allow force chain  propagation
    factor = 1.136 #float(input("Touch threshold: ")), change if desired
    touch = factor * d
    # angular radius (from apex) of the target region for force chains, decrease to allow chains further from apex to serve as force chain source 
    apex_deg = 5.0

    rows = []
    t0 = time.perf_counter()
    for N in range(N0, N1+1, step):
        phi = round((N/max_N)*0.861,4)
        print(f"\nphi = {phi}, Disks = {N}, ")
        pts_base, _, tags, s = hex_pack_for_N(R,h,d,N)
        succ=0
        contacts = []
        chain_counts = []  # Store chain counts for this N
        
        for _ in range(M):
            pts_r, theta_r = jitter_hex_pts(pts_base, s, d, R)
            adj            = build_graph_fast(pts_r, R, touch)
            edge_all       = find_edge_sets(pts_r, theta_r, R, h, d, tags)

            # Define upper ring: disks within apex_deg degrees of the apex
            upper_ring = set(np.where(theta_r <= math.radians(apex_deg))[0])
            
            # Search to ANY disk in upper_ring instead of single apex
            chain = find_chain_upward(adj, pts_r, list(edge_all), upper_ring, max_deg=90)
            
            # Count number of force chains (source disks reaching upper ring)
            num_chains = count_chain_sources(adj, pts_r, list(edge_all), upper_ring, max_deg=90)
            
            # Use 0 if no chain found, otherwise use the count
            if chain:
                succ += 1
                chain_counts.append(num_chains)
            else:
                chain_counts.append(0)
            
            interior = [i for i in range(N) if i not in edge_all]
            avg_c = np.mean([len(adj[i]) for i in interior]) if interior else 0.0
            contacts.append(avg_c)
        
        frac = succ/M
        meanc = float(np.mean(contacts))
        print(f"  Force chains per trial: {chain_counts}")
        chain_avg = np.average(chain_counts)
        chain_std = np.std(chain_counts)
        print(f"  Force chains average: {chain_avg}")
        print(f"  Force chains std: {chain_std}")
        print(f"  → success {succ}/{M}={frac:.3f}, mean contacts={meanc:.3f}")
        rows.append((N, phi, succ, M, frac, meanc))
    
    t1 = time.perf_counter()
    print(f"\nDone in {t1-t0:.2f}s")

    # writes data into .CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(script_dir, "results.csv")
    with open(fname,'w',newline='') as csv_file:
        w = csv.writer(csv_file)
        w.writerow(("N","phi","success_count","trials","success_fraction",
                    "mean_interior_contacts"))
        w.writerows(rows)
    print(f"Wrote data to {fname}")

    # quick summary plots
    phi   = [r[1] for r in rows]
    fracs = [r[4] for r in rows]
    plt.figure()
    plt.plot(phi, fracs, '-o') #Ns
    plt.xlabel('phi'); plt.ylabel('Chain success fraction')
    plt.grid(); plt.show()

if __name__=='__main__':
    main()
