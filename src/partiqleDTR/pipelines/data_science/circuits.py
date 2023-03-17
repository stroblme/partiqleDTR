import qiskit as q


def build_circuit_19(qc, n_qubits, identifier):
    for i in range(n_qubits):
        qc.rx(
            q.circuit.Parameter(f"{identifier}_rx_0_{i}"),
            i,
            f"{identifier}_rx_0_{i}",
        )
        qc.rz(q.circuit.Parameter(f"{identifier}_rz_1_{i}"), i)

    for i in range(n_qubits):
        if i == 0:
            qc.crx(
                q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"),
                n_qubits - 1,
                i,
                f"{identifier}_crx_{i+1}_{i}",
            )
        else:
            qc.crx(
                q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"),
                n_qubits - i - 1,
                n_qubits - i,
                f"{identifier}_crx_{i+1}_{i}",
            )

def build_circuit_19_missing(qc, n_qubits, identifier):
    for i in range(n_qubits):
        qc.rx(
            q.circuit.Parameter(f"{identifier}_rx_0_{i}"),
            i,
            f"{identifier}_rx_0_{i}",
        )
        qc.rz(q.circuit.Parameter(f"{identifier}_rz_1_{i}"), i)

    for i in range(n_qubits-1):
        if i == 0:
            qc.crx(
                q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"),
                i,
                n_qubits - 1,
                f"{identifier}_crx_{i+1}_{i}",
            )
        else:
            qc.crx(
                q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"),
                n_qubits - i,
                n_qubits - i - 1,
                f"{identifier}_crx_{i+1}_{i}",
            )


def build_circuit_19_flipped(qc, n_qubits, identifier):
    for i in range(n_qubits):
        qc.rx(
            q.circuit.Parameter(f"{identifier}_rx_0_{i}"),
            i,
            f"{identifier}_rx_0_{i}",
        )
        qc.rz(q.circuit.Parameter(f"{identifier}_rz_1_{i}"), i)

    for i in range(n_qubits):
        if i == 0:
            qc.crx(
                q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"),
                i,
                n_qubits - 1,
                f"{identifier}_crx_{i+1}_{i}",
            )
        else:
            qc.crx(
                q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"),
                n_qubits - i,
                n_qubits - i - 1,
                f"{identifier}_crx_{i+1}_{i}",
            )