import qiskit as q
import torch as t


def circuit_builder(
    qc, predefined_iec, predefined_vqc, n_qubits, n_layers, data_reupload=False
):
    # enc_params = gen_encoding_params(n_qubits, f"enc")

    try:
        pqc_generator = getattr(pqc_circuits, predefined_vqc)
    except AttributeError:
        print(
            f"Circuit {predefined_vqc} not found in {[m for m in dir(pqc_circuits) if not m.startswith('__')]}"
        )

    try:
        iec_generator = getattr(iec_circuits, predefined_iec)
    except AttributeError:
        print(
            f"Circuit {predefined_vqc} not found in {[m for m in dir(pqc_circuits) if not m.startswith('__')]}"
        )

    reuse_params = None
    for i in range(n_layers):
        if data_reupload or i == 0:
            reuse_params = iec_generator(qc, n_qubits, f"enc_{i}", reuse_params)
            qc.barrier()

        pqc_generator(qc, n_qubits, f"var_{i}")

        if i < n_layers - 1:
            qc.barrier()

    return qc


class iec_circuits:
    def direct_mapping(qc, n_qubits, identifier, reuse_params=None):
        energy_params = []
        prx_params = []
        pry_params = []
        prz_params = []

        for i in range(n_qubits):
            if reuse_params is not None:
                energy = reuse_params[0][i]
                prx = reuse_params[1][i]
                pry = reuse_params[2][i]
                prz = reuse_params[3][i]
            else:
                energy = q.circuit.Parameter(f"{identifier}_{i}_0")
                prx = q.circuit.Parameter(f"{identifier}_{i}_1")
                pry = q.circuit.Parameter(f"{identifier}_{i}_2")
                prz = q.circuit.Parameter(f"{identifier}_{i}_3")

            qc.rx(prx * energy * t.pi, i, f"{identifier}_rx_{i}")
            qc.ry(pry * energy * t.pi, i, f"{identifier}_ry_{i}")
            qc.rz(prz * energy * t.pi, i, f"{identifier}_rz_{i}")

            energy_params.append(energy)
            prx_params.append(prx)
            pry_params.append(pry)
            prz_params.append(prz)

        return [energy_params, prx_params, pry_params, prz_params]


class pqc_circuits:
    @staticmethod
    def variational(qc, n_qubits, identifier):
        for i in range(n_qubits):
            # if add_rot_gates:
            qc.rx(
                q.circuit.Parameter(f"{identifier}_rx_0_{i}"),
                i,
                f"{identifier}_rx_{i}",
            )
            qc.ry(
                q.circuit.Parameter(f"{identifier}_ry_0_{i}"),
                i,
                f"{identifier}_ry_{i}",
            )
            # qc.rz(
            #     q.circuit.Parameter(f"{identifier}_rz_0_{i}"),
            #     i,
            # )
            if i == 0:
                qc.crx(
                    q.circuit.Parameter(f"{identifier}_crx_{n_qubits - 1}_{i}"),
                    i,
                    n_qubits - 1,
                    f"{identifier}_crx_{n_qubits - 1}_{i}",
                )
                qc.cry(
                    q.circuit.Parameter(f"{identifier}_cry_{n_qubits - 1}_{i}"),
                    i,
                    n_qubits - 1,
                    f"{identifier}_cry_{n_qubits - 1}_{i}",
                )
                # qc.crz(
                #     q.circuit.Parameter(f"{identifier}_crz_{n_qubits - 1}_{i}"),
                #     i,
                #     n_qubits - 1,
                #     f"{identifier}_crz_{n_qubits - 1}_{i}",
                # )
            else:
                qc.crx(
                    q.circuit.Parameter(
                        f"{identifier}_crx_{n_qubits - i - 1}_{n_qubits - i}"
                    ),
                    n_qubits - i,
                    n_qubits - i - 1,
                    f"{identifier}_crx_{n_qubits - i - 1}_{n_qubits - i}",
                )
                qc.cry(
                    q.circuit.Parameter(
                        f"{identifier}_cry_{n_qubits - i - 1}_{n_qubits - i}"
                    ),
                    n_qubits - i,
                    n_qubits - i - 1,
                    f"{identifier}_cry_{n_qubits - i - 1}_{n_qubits - i}",
                )
                # qc.crz(
                #     q.circuit.Parameter(f"{identifier}_crz_{n_qubits - i - 1}_{n_qubits - i}"),
                #     n_qubits - i,
                #     n_qubits - i - 1,
                #     f"{identifier}_crz_{n_qubits - i - 1}_{n_qubits - i}",
                # )

    @staticmethod
    def circuit_19(qc, n_qubits, identifier):
        """
        Original circuit 19 implementation

        Args:
            qc (_type_): _description_
            n_qubits (_type_): _description_
            identifier (_type_): _description_
        """
        for i in range(n_qubits):
            qc.rx(
                q.circuit.Parameter(f"{identifier}_rx_0_{i}"),
                i,
                f"{identifier}_rx_0_{i}",
            )
            qc.rz(
                q.circuit.Parameter(f"{identifier}_rz_1_{i}"),
                i,
                f"{identifier}_rz_1_{i}",
            )

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

    @staticmethod
    def circuit_191(qc, n_qubits, identifier):
        """
        Circuit 19 with missing last CRX gate

        Args:
            qc (_type_): _description_
            n_qubits (_type_): _description_
            identifier (_type_): _description_
        """
        for i in range(n_qubits):
            qc.rx(
                q.circuit.Parameter(f"{identifier}_rx_0_{i}"),
                i,
                f"{identifier}_rx_0_{i}",
            )
            qc.rz(
                q.circuit.Parameter(f"{identifier}_rz_1_{i}"),
                i,
                f"{identifier}_rz_1_{i}",
            )

        for i in range(n_qubits - 1):
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

    @staticmethod
    def circuit_192(qc, n_qubits, identifier):
        """
        Circuit 19 flipped

        Args:
            qc (_type_): _description_
            n_qubits (_type_): _description_
            identifier (_type_): _description_
        """
        for i in range(n_qubits):
            qc.rx(
                q.circuit.Parameter(f"{identifier}_rx_0_{i}"),
                i,
                f"{identifier}_rx_0_{i}",
            )
            qc.rz(
                q.circuit.Parameter(f"{identifier}_rz_1_{i}"),
                i,
                f"{identifier}_rz_1_{i}",
            )

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

    @staticmethod
    def circuit_18(qc, n_qubits, identifier):
        for i in range(n_qubits):
            qc.rx(
                q.circuit.Parameter(f"{identifier}_rx_0_{i}"),
                i,
                f"{identifier}_rx_0_{i}",
            )
            qc.rz(
                q.circuit.Parameter(f"{identifier}_rz_1_{i}"),
                i,
                f"{identifier}_rz_1_{i}",
            )

        for i in range(n_qubits):
            if i == 0:
                qc.crz(
                    q.circuit.Parameter(f"{identifier}_crz_{i+1}_{i}"),
                    n_qubits - 1,
                    i,
                    f"{identifier}_crz_{i+1}_{i}",
                )
            else:
                qc.crz(
                    q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"),
                    n_qubits - i - 1,
                    n_qubits - i,
                    f"{identifier}_crz_{i+1}_{i}",
                )
