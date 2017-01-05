#!/usr/bin/env python
"""Provides conversion for molecules."""

import openbabel as ob
import pybel
import networkx as nx
from eden.util import read
import time
import datetime
import logging
logger = logging.getLogger(__name__)


def load(data, file_format='sdf'):
    """load."""
    start = time.time()
    iterable = mol_file_to_iterable(filename=data,
                                    file_format=file_format)
    graphs = _obabel_to_eden(iterable, file_format=file_format)
    for graph in graphs:
        yield graph
    delta_time = datetime.timedelta(seconds=(time.time() - start))
    logger.debug('Elapsed time: %s' % (str(delta_time)))


def mol_file_to_iterable(filename=None, file_format=None):
    """Parse multiline file into text blocks."""
    if file_format == 'sdf':
        with open(filename) as f:
            s = ''
            for line in f:
                if line.strip() != '$$$$':
                    s = s + line
                else:
                    return_value = s + line
                    s = ''
                    yield return_value
    elif file_format == 'smi':
        with open(filename) as f:
            for line in f:
                yield line
    else:
        raise Exception('ERROR: unrecognized file format: %s' % file_format)


def _smi_has_error(smi):
    smi = smi.strip()
    n_open_parenthesis = sum(1 for c in smi if c == '(')
    n_close_parenthesis = sum(1 for c in smi if c == ')')
    n_open_parenthesis_square = sum(1 for c in smi if c == '[')
    n_close_parenthesis_square = sum(1 for c in smi if c == ']')
    return (n_open_parenthesis != n_close_parenthesis) or \
        (n_open_parenthesis_square != n_close_parenthesis_square)


def _obabel_to_eden(iterable, file_format=None):
    counter = 0
    if file_format == 'sdf':
        for graph in _sdf_to_eden(iterable):
            counter += 1
            yield graph
    elif file_format == 'smi':
        for graph in _smi_to_eden(iterable):
            counter += 1
            yield graph
    else:
        raise Exception('ERROR: unrecognized file format: %s' %
                        file_format)
    logger.debug('Converted %d molecules' % counter)


def _sdf_to_eden(iterable):
    for mol_sdf in read(iterable):
        mol = pybel.readstring("sdf", mol_sdf.strip())
        # remove hydrogens
        mol.removeh()
        graph = _obabel_to_networkx(mol)
        if len(graph):
            yield graph


def _smi_to_eden(iterable):
    for mol_smi in read(iterable):
        if _smi_has_error(mol_smi) is False:
            mol = pybel.readstring("smi", mol_smi.strip())
            # remove hydrogens
            mol.removeh()
            graph = _obabel_to_networkx(mol)
            if len(graph):
                graph.graph['id'] = mol_smi.strip()
                yield graph


def _obabel_to_networkx(mol):
    """Take a pybel molecule object and converts it into a graph."""
    graph = nx.Graph()
    # atoms
    for atom in mol:
        node_id = atom.idx - 1
        label = str(atom.type)
        graph.add_node(node_id, label=label)
    # bonds
    for bond in ob.OBMolBondIter(mol.OBMol):
        label = str(bond.GetBO())
        graph.add_edge(
            bond.GetBeginAtomIdx() - 1,
            bond.GetEndAtomIdx() - 1,
            label=label)
    return graph
