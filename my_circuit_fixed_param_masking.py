# Copyright 2021-2022 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Circuit Ansatz
==============
A circuit ansatz converts a DisCoCat diagram into a quantum circuit.

"""
from __future__ import annotations

__all__ = ['CircuitAnsatz', 'IQPAnsatz']

from abc import abstractmethod
from collections.abc import Mapping
from itertools import cycle
from typing import Callable, Optional

from discopy.quantum.circuit import (Circuit, Discard, Functor, Id,
                                     IQPansatz as IQP, qubit,
                                     Sim14ansatz as Sim14,
                                     Sim15ansatz as Sim15)
from discopy.quantum.gates import Bra, H, Ket, Rx, Ry, Rz, CX, Controlled
from discopy.rigid import Box, Diagram, Ty
import numpy as np
from sympy import Symbol, symbols

from lambeq.ansatz import BaseAnsatz
import random
computational_basis = Id(qubit)


class CircuitAnsatz(BaseAnsatz):
    """Base class for circuit ansatz."""

    def __init__(self,
                 masked_token: str,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int,
                 circuit: Callable[[int, np.ndarray], Circuit],
                 discard: bool = False,
                 single_qubit_rotations: Optional[list[Circuit]] = None,
                 postselection_basis: Circuit = computational_basis,
                 ) -> None:
        """Instantiate a circuit ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.rigid.Ty` to the number of
            qubits it uses in a circuit.
        n_layers : int
            The number of layers used by the ansatz.
        n_single_qubit_params : int
            The number of single qubit rotations used by the ansatz.
        circuit : callable
            Circuit generator used by the ansatz. This is a function
            (or a class constructor) that takes a number of qubits and
            a numpy array of parameters, and returns the ansatz of that
            size, with parameterised boxes.
        discard : bool, default: False
            Discard open wires instead of post-selecting.
        postselection_basis: Circuit, default: Id(qubit)
            Basis to post-select in, by default the computational basis.
        single_qubit_rotations: list of Circuit, optional
            The rotations to be used for a single qubit. When only a
            single qubit is present, the ansatz defaults to applying a
            series of rotations in a cycle, determined by this parameter
            and `n_single_qubit_params`.

        """
        self.ob_map = ob_map
        self.n_layers = n_layers
        self.n_single_qubit_params = n_single_qubit_params
        self.circuit = circuit
        self.discard = discard
        self.postselection_basis = postselection_basis
        self.single_qubit_rotations = single_qubit_rotations or []
        self.masked_token = masked_token

        self.functor = Functor(ob=ob_map, ar=self._ar)

    def __call__(self, diagram: Diagram) -> Circuit:
        """Convert a DisCoPy diagram into a DisCoPy circuit."""
        return self.functor(diagram)

    def ob_size(self, pg_type: Ty) -> int:
        """Calculate the number of qubits used for a given type."""
        return sum(self.ob_map[Ty(factor.name)] for factor in pg_type)

    @abstractmethod
    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        """Calculate the shape of the parameters required."""

    def _ar(self, box: Box) -> Circuit:
        label = self._summarise_box(box)
        dom, cod = self.ob_size(box.dom), self.ob_size(box.cod)

        # For masking tokens
        if box.name == self.masked_token:
            n_qubits = max(dom, cod)
            if n_qubits == 0:
                circuit = Id()

            elif n_qubits == 1:
                sum_value = 0
                circuit = Rx(random.random())
                for gate in cycle(['Rz', 'Rx']):
                    if gate == 'Rz':
                        circuit >>= Rz(random.random())
                    elif gate == 'Rx':
                        circuit >>= Rx(random.random())
                    sum_value += 1
                    if sum_value >= self.n_single_qubit_params-1:
                        break

            elif n_qubits == 2:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0]

                for _ in range(self.n_layers):
                    circuit >>= H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0]

            elif n_qubits == 3:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1]



            elif n_qubits == 4:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2]


            elif n_qubits == 5:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3]


            elif n_qubits == 6:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4]

            elif n_qubits == 7:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5]



            elif n_qubits == 8:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6]



            elif n_qubits == 9:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7]


            elif n_qubits == 10:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ gate_sequence[8]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ gate_sequence[8]


            elif n_qubits == 11:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ gate_sequence[8] @\
                                gate_sequence[9]



                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ gate_sequence[8] @\
                                    gate_sequence[9]

            elif n_qubits == 12:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                    gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10]

            elif n_qubits == 13:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                    gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11]

            elif n_qubits == 14:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                    gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]

            elif n_qubits == 15:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                @ gate_sequence[13]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                    gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                    @ gate_sequence[13]

            elif n_qubits == 16:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                @ gate_sequence[13] @ gate_sequence[14]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                    gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                    @ gate_sequence[13] @ gate_sequence[14]


            elif n_qubits == 17:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                @ gate_sequence[13] @ gate_sequence[14] @ gate_sequence[15]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                    gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                    @ gate_sequence[13] @ gate_sequence[14] @ gate_sequence[15]

            elif n_qubits == 18:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                @ gate_sequence[13] @ gate_sequence[14] @ gate_sequence[15] @ gate_sequence[16]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                    gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                    @ gate_sequence[13] @ gate_sequence[14] @ gate_sequence[15] @ gate_sequence[16]

            elif n_qubits == 19:
                circuit = H
                for _ in range(n_qubits - 1):
                    circuit @= H
                for i in range(n_qubits - 1):
                    gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                    gate_sequence[i] = Controlled(Rz(random.random()))
                    circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                @ gate_sequence[13] @ gate_sequence[14] @ gate_sequence[15] @ gate_sequence[16] @ gate_sequence[17]

                for _ in range(self.n_layers):
                    circuit >>= H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H @ H
                    for i in range(n_qubits - 1):
                        gate_sequence = [Id(qubit) for _ in range(n_qubits - 1)]
                        gate_sequence[i] = Controlled(Rz(random.random()))
                        circuit >>= gate_sequence[0] @ gate_sequence[1] @ gate_sequence[2] @ gate_sequence[3] @ \
                                    gate_sequence[4] @ gate_sequence[5] @ gate_sequence[6] @ gate_sequence[7] @ \
                                    gate_sequence[8] @ gate_sequence[9] @ gate_sequence[10] @ gate_sequence[11] @ gate_sequence[12]\
                                    @ gate_sequence[13] @ gate_sequence[14] @ gate_sequence[15] @ gate_sequence[16] @ gate_sequence[17]
        # For normal tokens
        else:
            n_qubits = max(dom, cod)
            if n_qubits == 0:
                circuit = Id()
            elif n_qubits == 1:
                syms = symbols(f'{label}_0:{self.n_single_qubit_params}',
                               cls=Symbol)
                circuit = Id(qubit)
                for rot, sym in zip(cycle(self.single_qubit_rotations), syms):
                    circuit >>= rot(sym)
            else:
                params_shape = self.params_shape(n_qubits)
                syms = symbols(f'{label}_0:{np.prod(params_shape)}', cls=Symbol)
                params: np.ndarray = np.array(syms).reshape(params_shape)
                circuit = self.circuit(n_qubits, params)

        if cod > dom:
            circuit <<= Id(dom) @ Ket(*[0]*(cod - dom))
        elif self.discard:
            circuit >>= Id(cod) @ Discard(dom - cod)
        else:
            circuit >>= Id(cod).tensor(*[self.postselection_basis] * (dom-cod))
            circuit >>= Id(cod) @ Bra(*[0]*(dom - cod))
        return circuit


class IQPAnsatz(CircuitAnsatz):
    """Instantaneous Quantum Polynomial ansatz.

    An IQP ansatz interleaves layers of Hadamard gates with diagonal
    unitaries. This class uses :py:obj:`n_layers-1` adjacent CRz gates
    to implement each diagonal unitary.

    """

    def __init__(self,
                 masked_token,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 discard: bool = False) -> None:
        """Instantiate an IQP ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.rigid.Ty` to the number of
            qubits it uses in a circuit.
        n_layers : int
            The number of layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
        discard : bool, default: False
            Discard open wires instead of post-selecting.
            :param masked_token:

        """
        super().__init__(masked_token,
                         ob_map,
                         n_layers,
                         n_single_qubit_params,
                         IQP,
                         discard,
                         [Rx, Rz],
                         H)

    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, n_qubits - 1)
