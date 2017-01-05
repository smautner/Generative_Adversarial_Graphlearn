#!/usr/bin/env python
"""Provides io for molecules."""

import os
import requests
import logging
logger = logging.getLogger(__name__)


root_uri = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/'


def get_assay_description(assay_id):
    """get_assay_description."""
    query = root_uri
    query += 'assay/aid/%s/summary/JSON' % assay_id
    reply = requests.get(query)
    name = reply.json()['AssaySummaries']['AssaySummary'][0]['Name']
    return name


def _get_compounds(fname, size, listkey, stepsize=50):
    with open(fname, 'w') as file_handle:
        index_start = 0
        for chunk, index_end in enumerate(range(0, size + stepsize, stepsize)):
            if index_end is not 0:
                t = 'Chunk %s) Processing compounds %s to %s (%s)' % \
                    (chunk, index_start, index_end - 1, size)
                logger.debug(t)
                query = root_uri
                query += 'compound/listkey/' + str(listkey)
                query += '/SDF?&listkey_start=' + str(index_start)
                query += '&listkey_count=' + str(stepsize)
                reply = requests.get(query)
                file_handle.write(reply.text)
            index_start = index_end
        print 'compounds available in file: ', fname


def _make_rest_query(assay_id, active=True):
    if active:
        mode = 'active'
    else:
        mode = 'inactive'
    core = 'assay/aid/%s/cids/JSON?cids_type=%s&list_return=listkey' % \
        (assay_id, mode)
    rest_query = root_uri + core
    return rest_query


def _query_db(assay_id, fname=None, active=True, stepsize=50):
    reply = requests.get(_make_rest_query(assay_id, active=active))
    # extract the listkey
    listkey = reply.json()['IdentifierList']['ListKey']
    size = reply.json()['IdentifierList']['Size']
    _get_compounds(fname=fname, size=size, listkey=listkey, stepsize=stepsize)


def download(assay_id, active=True, stepsize=50):
    """download."""
    pubchem_dir = 'PUBCHEM'
    if not os.path.exists(pubchem_dir):
        os.mkdir(pubchem_dir)
    if active:
        fname = 'AID%s_active.sdf' % assay_id
    else:
        fname = 'AID%s_inactive.sdf' % assay_id
    full_fname = os.path.join(pubchem_dir, fname)
    if not os.path.isfile(full_fname):
        logger.debug('Querying PubChem for AID: %s' % assay_id)
        _query_db(assay_id,
                  fname=full_fname,
                  active=active,
                  stepsize=stepsize)
    else:
        logger.debug('Reading from file: %s' % full_fname)
    return full_fname
