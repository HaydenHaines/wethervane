"""Bedrock model data contracts.

Two-layer architecture:
  Layer 1 — Communities: geographically contiguous blobs (hard assignment)
  Layer 2 — Types: abstract archetypes with soft membership weights

A community in rural Georgia and one in rural Washington are DIFFERENT communities.
They may both be TYPE 3 ("Consolidating Rural Red") but they are distinct geographic blobs.
"""
