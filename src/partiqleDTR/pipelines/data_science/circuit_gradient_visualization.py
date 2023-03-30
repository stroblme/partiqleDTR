from qiskit.visualization.circuit.matplotlib import MatplotlibDrawer as qiskit_matplotlibdrawer
from qiskit.visualization.circuit.circuit_visualization import _utils
from qiskit.dagcircuit.dagnode import DAGOpNode

import torch as t
import plotly

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b).upper()


def draw_gradient_circuit(
    gradients,
    circuit,
    scale=None,
    filename=None,
    style=None,
    plot_barriers=True,
    reverse_bits=False,
    justify=None,
    idle_wires=True,
    with_layout=True,
    fold=None,
    ax=None,
    initial_state=False,
    cregbundle=None,
    wire_order=None,
):
    if type(gradients) == list:
        gradients = t.stack(gradients)

    # first mean over epochs, then abs value of gradient
    gradients_mean = gradients.mean(axis=0).abs()
    
    fig = draw_single_gradient_circuit(
            gradients_mean,
            circuit,
            scale=scale,
            filename=filename,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            fold=fold,
            ax=ax,
            initial_state=initial_state,
            cregbundle=cregbundle,
            wire_order=wire_order
        )
    
    return fig



# class DAGOpNode(DAGOpNode):
#     _significance = 1

#     @property
#     def significance(self):
#         return self.significance
#     @significance.setter
#     def significance(self, value):
#         self._significance = value
#     @significance.deleter
#     def significance(self):
#         del self._significance


def draw_single_gradient_circuit(
    significance,
    circuit,
    scale=None,
    filename=None,
    style=None,
    plot_barriers=True,
    reverse_bits=False,
    justify=None,
    idle_wires=True,
    with_layout=True,
    fold=None,
    ax=None,
    initial_state=False,
    cregbundle=None,
    wire_order=None,
):
    qubits, clbits, nodes = _utils._get_layered_instructions(
        circuit,
        reverse_bits=reverse_bits,
        justify=justify,
        idle_wires=idle_wires,
        wire_order=wire_order,
    )
    if fold is None:
        fold = 25
        
    qcd = custom_matplotlib(
        qubits,
        clbits,
        nodes,
        scale=scale,
        style=style,
        reverse_bits=reverse_bits,
        plot_barriers=plot_barriers,
        layout=None,
        fold=fold,
        ax=ax,
        initial_state=initial_state,
        cregbundle=cregbundle,
        global_phase=None,
        calibrations=None,
        qregs=None,
        cregs=None,
        with_layout=with_layout,
        circuit=circuit,
    )

    label_map = {}
    i = 0
    for p in circuit.parameters:
        if "var" in p.name:
            label_map[p.name] = i
            i += 1

    qcd.set_node_significance(label_map, significance)

    return qcd.draw(filename)


class custom_matplotlib(qiskit_matplotlibdrawer):
    nodes_significance = {}

    def set_node_significance(self, label_map, significance):
        max_sig = max(significance)
        min_sig = min(significance)

        i = 0
        for layer in self._nodes:
            for node in layer:
                if node.op.label is not None and "var" in node.op.label:
                    self.nodes_significance[node] = ((significance[label_map[node.op.label]] - min_sig) / max_sig).item()
                    i += 1

  
   
    def _get_colors(self, node):
        """Get all the colors needed for drawing the circuit"""
        op = node.op
        base_name = None if not hasattr(op, "base_gate") else op.base_gate.name
        color = None
        if self._data[node]["raw_gate_text"] in self._style["dispcol"]:
            color = self._style["dispcol"][self._data[node]["raw_gate_text"]]
        elif op.name in self._style["dispcol"]:
            color = self._style["dispcol"][op.name]

        if node in self.nodes_significance:
            c = plotly.colors.sample_colorscale(plotly.colors.sequential.Bluered, samplepoints=self.nodes_significance[node], colortype='tuple')

            color = (rgb_to_hex(r=int(c[0][0]*255), g=int(c[0][1]*255), b=int(c[0][2]*255)), rgb_to_hex(r=int(255-c[0][0]*255), g=int(255-c[0][1]*255), b=int(255-c[0][2]*255)))

        if color is not None:
            # Backward compatibility for style dict using 'displaycolor' with
            # gate color and no text color, so test for str first
            if isinstance(color, str):
                fc = color
                gt = self._style["gt"]
            else:
                fc = color[0]
                gt = color[1]
        # Treat special case of classical gates in iqx style by making all
        # controlled gates of x, dcx, and swap the classical gate color
        elif self._style["name"] in ["iqx", "iqx-dark"] and base_name in ["x", "dcx", "swap"]:
            color = self._style["dispcol"][base_name]
            if isinstance(color, str):
                fc = color
                gt = self._style["gt"]
            else:
                fc = color[0]
                gt = color[1]
        else:
            fc = self._style["gc"]
            gt = self._style["gt"]

        if self._style["name"] == "bw":
            ec = self._style["ec"]
            lc = self._style["lc"]
        else:
            ec = fc
            lc = fc
        # Subtext needs to be same color as gate text
        sc = gt
        self._data[node]["fc"] = fc
        self._data[node]["ec"] = ec
        self._data[node]["gt"] = gt
        self._data[node]["tc"] = self._style["tc"]
        self._data[node]["sc"] = sc
        self._data[node]["lc"] = lc
