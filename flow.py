import numpy as np
import matplotlib.pyplot as plt

class MaximalFlow:
    def __init__(self, vertices, capacity):
        self.vertices = {key:np.asarray(value) for key, value in vertices.items()}
        self.residuals = capacity.copy()
        self.capacity = capacity.copy()
        self.flow = {key:0 for key in capacity.keys()}
        self.TEXT_OFFSET = 0.035
        self.FIG_SIZE = (20, 7)
        self.ax = None
    
    # ------------------------FINDING MAXIMAL FLOW------------------------
    def find_maximal_flow(self, draw_results=True, show_intermediate=False):
        if show_intermediate: self.draw_figure(flow_title='Flow network', residual_title='Residual network')
        while self.find_st_path() is not None:
            self.update_flows(self.find_st_path())
            if show_intermediate: self.draw_figure()
        if draw_results and not show_intermediate: self.draw_figure(flow_title='Flow network', residual_title='Residual network')
    
    def update_flows(self, path):
        f = min([self.residuals[(a, b)] for a, b in zip(path[:-1], path[1:])])
        for a, b in zip(path[:-1], path[1:]):
            self.residuals[(a, b)] -= f
            if self.residuals[(a, b)] == 0: self.residuals.pop((a, b), None)
            if (a, b) in set(self.flow.keys()):
                self.flow[(a, b)] += f
                self.residuals[(b, a)] = self.flow[(a, b)]
            else:
                self.flow[(b, a)] -= f
                self.residuals[(b, a)] += f
    
    def find_st_path(self):
        path = self.path_to_t('s', ['s'])
        if len(path) > 0 and path[-1] == 't': return path
        return None
    
    def path_to_t(self, x, acc):
        path = []
        for a, b in self.residuals.keys():
            if a == x:
                if b == 't': return acc + [b]
                elif b not in acc: path = self.path_to_t(b, acc + [b])
            if len(path) > 0 and path[-1] == 't': return path
        return path
    
    # ------------------------FINDING MINIMAL CUT------------------------
    def get_minimal_cut(self):
        A = self.get_vertices_from('s', {'s'})
        B = set(self.vertices.keys()).difference(A)
        return A, B
    
    def get_vertices_from(self, vertex, acc):
        for a, b in self.residuals:
            if a == vertex and b not in acc:
                acc = acc.union(b)
                acc = acc.union(self.get_vertices_from(b, acc))
        return acc
    
    def get_minimal_cut_value(self):
        A, B = self.get_minimal_cut()
        return sum([self.capacity[(a, b)] for a in A for b in B if (a, b) in set(self.capacity.keys())])
    
    # ------------------------DRAWING FIGURES------------------------
    def draw_figure(self, flow_title=None, residual_title=None):
        _, axs = plt.subplots(1, 2, figsize=self.FIG_SIZE)
        for ax, title, edges in zip(axs, [flow_title, residual_title], [self.flow, self.residuals]):
            self.ax = ax
            self.ax.axis('off')
            self.ax.set_title(title, size=20)
            self.draw_graph(edges)
        plt.show()
    
    def draw_graph(self, edges):
        self.draw_vertices()
        self.draw_edges(edges)

    def draw_vertices(self):
        self.ax.scatter(*np.asarray(list(self.vertices.values())).T, edgecolors='black', s=300, facecolors='none')
        self.label_vertices()

    def label_vertices(self):
        for label, coordinates in self.vertices.items():
            self.ax.annotate(label, coordinates - self.TEXT_OFFSET)

    def draw_edges(self, edges):
        for (a, b), capacity in edges.items():
            edge, margin, orthogonal = self.get_edge_parameters(a, b)
            origin = self.vertices[a] + margin if (b, a) not in set(edges.keys())\
                else self.vertices[a] + margin + 0.3 * orthogonal
            self.ax.arrow(*origin, *(edge - 2 * margin), color='black', width=0.012, length_includes_head=True)
            self.ax.annotate(capacity, self.vertices[a] + 2.5 * margin + orthogonal - self.TEXT_OFFSET)
    
    def get_edge_parameters(self, a, b):
        edge = self.vertices[b] - self.vertices[a]
        margin = edge/np.linalg.norm(edge) * 0.2
        orthogonal = self.get_orthonormal(edge) * 0.1
        return edge, margin, orthogonal
    
    @staticmethod
    def get_orthonormal(vector):
        return np.asarray([-vector[1], vector[0]])/np.linalg.norm(np.asarray([-vector[1], vector[0]]))