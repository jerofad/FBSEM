"""
Scanner parameter dictionaries for supported PET systems.

Add a new scanner by defining a new dict below with the same keys and
passing its name to BuildGeometry_v4.
"""

# Siemens mCT (model 1104)
mct = {
    "model_number":          1104,
    "circularGantry":        1,
    "nBuckets":              48,
    "nBlockRings":           4,
    "nBlockPerRing":         48,
    "nPhysCrystalsPerBlock": 13,
    "useVirtualCrystal":     1,
    "detectorRadiusCm":      42.76,
    "sinogramDOIcm":         0.67,
    "LORDOIcm":              0.96,
    "nRadialBins":           400,
    "nMash":                 2,
    "rCrystalDimCm":         2.0,
    "xCrystalDimCm":         0.40728,
    "zCrystalDimCm":         0.4050,
    "transaxialFovCm":       69.7266,
    "span":                  11,
    "nSegments":             9,
    "maxRingDiff":           49,
    "nTofBins":              13,
    "coinciWindowWidthNsec": 4.0625,
    "tofResolutionNsec":     0.580,
    "tofOffsetNsec":         0.039,
}

# Siemens mMR (model 2008)
mmr = {
    "model_number":          2008,
    "circularGantry":        1,
    "nBuckets":              224,
    "nBlockRings":           8,
    "nBlockPerRing":         56,
    "nPhysCrystalsPerBlock": 8,
    "useVirtualCrystal":     1,
    "detectorRadiusCm":      32.8,
    "sinogramDOIcm":         0.67,
    "LORDOIcm":              0.96,
    "nRadialBins":           344,
    "nMash":                 1,
    "rCrystalDimCm":         2.0,
    "xCrystalDimCm":         0.41725,
    "zCrystalDimCm":         0.40625,
    "transaxialFovCm":       60.0,
    "span":                  11,
    "nSegments":             11,
    "maxRingDiff":           60,
    "nTofBins":              1,
    "coinciWindowWidthNsec": 5.85938,
    "tofResolutionNsec":     5.85938,
    "tofOffsetNsec":         0,
}

SUPPORTED_SCANNERS = {"mmr": mmr, "mct": mct}
