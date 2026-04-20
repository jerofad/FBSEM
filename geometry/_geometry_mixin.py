"""
Scanner geometry mixin for BuildGeometry_v4.

Responsibilities:
  - Load scanner parameters into self.scanner / self.sinogram / self.image structs
  - Compute LOR end-point coordinates (transaxial and axial)
  - Build the Michelogram (axial sinogram plane layout)
  - Compute / load the system matrix (Siddon ray tracer)
"""

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


class _GeometryMixin:
    """Scanner setup, LOR geometry, and system-matrix I/O."""

    # ── Scanner initialisation ─────────────────────────────────────────────────

    def _computeGantryInfo(self, g: dict) -> None:
        """Populate self.scanner / self.sinogram / self.image from a scanner dict."""
        self._load_gantry_dict(g)
        sc = self.scanner
        si = self.sinogram

        sc.nCrystalsPerBlock = sc.nPhysCrystalsPerBlock + sc.useVirtualCrystal
        sc.nCrystalsPerRing  = sc.nBlockPerRing * sc.nCrystalsPerBlock

        if sc.model_number == 1104:   # mCT
            sc.nCrystalRings = (sc.nBlockRings * sc.nPhysCrystalsPerBlock
                                + (sc.nBlockRings - 1) * sc.useVirtualCrystal)
        elif sc.model_number == 2008:  # mMR
            sc.nCrystalRings = sc.nBlockRings * sc.nPhysCrystalsPerBlock

        sc.effDetectorRadiusCm  = sc.detectorRadiusCm + sc.LORDOIcm
        sc.isTof                = si.nTofBins > 1
        sc.TofBinWidthNsec      = sc.coinciWindowWidthNsec / si.nTofBins
        sc.planeSepCm           = sc.zCrystalDimCm / 2.0
        si.nAngularBins         = sc.nCrystalsPerRing // 2 // si.nMash

        self.image.matrixSize  = [si.nRadialBins, si.nRadialBins, 2 * sc.nCrystalRings - 1]
        self.image.voxelSizeCm = [sc.xCrystalDimCm / 2.0,
                                  sc.xCrystalDimCm / 2.0,
                                  sc.planeSepCm]
        self.is3d = False

    def _load_gantry_dict(self, g: dict) -> None:
        """Copy raw scanner dict values into self.scanner / self.sinogram."""
        sc = self.scanner
        si = self.sinogram

        sc.model_number          = g["model_number"]
        sc.circularGantry        = g["circularGantry"]
        sc.nBuckets              = g["nBuckets"]
        sc.nBlockRings           = g["nBlockRings"]
        sc.nBlockPerRing         = g["nBlockPerRing"]
        sc.nPhysCrystalsPerBlock = g["nPhysCrystalsPerBlock"]
        sc.useVirtualCrystal     = g["useVirtualCrystal"]
        sc.detectorRadiusCm      = g["detectorRadiusCm"]
        sc.sinogramDOIcm         = g["sinogramDOIcm"]
        sc.LORDOIcm              = g["LORDOIcm"]
        sc.rCrystalDimCm         = g["rCrystalDimCm"]
        sc.xCrystalDimCm         = g["xCrystalDimCm"]
        sc.zCrystalDimCm         = g["zCrystalDimCm"]
        sc.transaxialFovCm       = g["transaxialFovCm"]
        sc.maxRingDiff           = g["maxRingDiff"]
        sc.coinciWindowWidthNsec = g["coinciWindowWidthNsec"]
        sc.tofResolutionNsec     = g["tofResolutionNsec"]
        sc.tofOffsetNsec         = g["tofOffsetNsec"]

        cf = si.radialBinCropfactor
        si.nRadialBins_orig = g["nRadialBins"]
        si.nRadialBins      = g["nRadialBins"] - int(np.ceil(g["nRadialBins"] * cf / 2.0) * 2)
        si.nMash            = g["nMash"]
        si.span             = g["span"]
        si.nSegments        = g["nSegments"]
        si.nTofBins         = g["nTofBins"]

    def setTo3d(self) -> None:
        """Switch the geometry object to 3-D mode (resets FOV mask)."""
        self.is3d     = True
        self.fov_mask = None

    def setApirlMmrEngine(self, binPath: str = None, temPath: str = None,
                          gpu: bool = True, multiprocess: bool = True) -> None:
        """Configure the APIRL external reconstruction engine (3-D only)."""
        if binPath is None:
            binPath = r"C:\MatlabWorkSpace\apirl-tags\APIRL1.3.3_win64_cuda8.0_sm35\build\bin"
        if temPath is None:
            temPath = os.getcwd()
        os.makedirs(temPath, exist_ok=True)

        self.engine.temPath    = temPath
        self.engine.binPath    = binPath
        self.engine.gpu        = gpu
        self.engine.multiprocess = multiprocess

        self.setTo3d()
        self.buildMichelogram()
        self.sinogram.shape = [
            self.sinogram.nRadialBins,
            self.sinogram.nAngularBins,
            self.sinogram.totalNumberOfSinogramPlanes,
        ]

    # ── Michelogram ────────────────────────────────────────────────────────────

    def buildMichelogram(self):
        """Compute the axial sinogram plane layout (Michelogram).

        Returns a list of arrays, one per segment, where each element gives
        the crystal-ring pair indices forming each sinogram plane.
        """
        nRings = self.scanner.nCrystalRings
        a      = np.arange(1, nRings ** 2 + 1).reshape(nRings, nRings).T
        b      = np.arange(-self.scanner.maxRingDiff, self.scanner.maxRingDiff + 1).reshape(
                     self.sinogram.nSegments, self.sinogram.span)
        mid    = self.sinogram.nSegments // 2
        isodd  = np.remainder(b[mid, 0], 2)
        Segments = []
        maxPlanesPerSeg = np.zeros([self.sinogram.nSegments, 2], dtype="int16")

        for j in range(self.sinogram.nSegments):
            diags = [np.diag(a, k=b[j, i]) for i in range(self.sinogram.span)]
            if j == mid and isodd:
                c, k = 0, 1
            else:
                c, k = 1, 0
            odd,  maxPlanesPerSeg[j, 0] = self._zero_pad(diags[k::2])
            even, maxPlanesPerSeg[j, 1] = self._zero_pad(diags[c::2])
            n_planes = int(np.sum(maxPlanesPerSeg[j, :]))
            planes   = np.empty((n_planes,), dtype=object)
            planes[0::2] = self._zero_trim(odd)
            planes[1::2] = self._zero_trim(even)
            Segments.append(planes)

        self.sinogram.numberOfPlanesPerSeg      = np.sum(maxPlanesPerSeg, axis=1)
        self.sinogram.totalNumberOfSinogramPlanes = int(np.sum(self.sinogram.numberOfPlanesPerSeg))
        return Segments

    def plotMichelogram(self, showRingNumber: bool = False) -> None:
        """Plot the Michelogram using matplotlib."""
        from matplotlib import pyplot as plt
        Segments = self.buildMichelogram()
        nRings   = self.scanner.nCrystalRings
        nS       = self.sinogram.nSegments
        grid     = np.zeros([nRings ** 2, 1], dtype="int16")
        colours  = np.concatenate([np.arange(0, (nS - 1) / 2),
                                    [(nS - 1) / 2 + 1],
                                    np.arange((nS - 1) / 2 - 1, -1, -1)]) + 1
        for i in range(nS):
            idx = np.concatenate(Segments[i][:]) - 1
            grid[idx] = colours[i]
        grid = grid.reshape([nRings, nRings])

        plt.imshow(grid, aspect="equal")
        if showRingNumber:
            for k, (j, i) in enumerate(np.ndindex(grid.shape)):
                plt.text(i, j, str(k + 1), ha="center", va="center", fontsize=12)
        ax = plt.gca()
        ticks = np.arange(0, nRings)
        ax.set_xticks(ticks); ax.set_xticklabels(ticks + 1)
        ax.set_yticks(ticks); ax.set_yticklabels(ticks + 1)
        ax.set_xticks(np.arange(-0.5, nRings), minor=True)
        ax.set_yticks(np.arange(-0.5, nRings), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=2)
        plt.title(f"Michelogram — span={self.sinogram.span}, nSegments={self.sinogram.nSegments}",
                  fontsize=18)
        plt.tight_layout()
        plt.show()

    # ── LOR coordinate computation ─────────────────────────────────────────────

    def LorsAxialCoor(self):
        """Compute axial (z) coordinates of LOR end-points for each segment.

        Returns
        -------
        axialCoorPerSeg : list of (nPlanes, 4) arrays [z1, z2, r, -r] per segment
        z_axis          : 1-D array of crystal ring z positions (cm)
        """
        Segments = self.buildMichelogram()
        z_axis   = (self.scanner.zCrystalDimCm
                    * np.arange(-(self.scanner.nCrystalRings - 1) / 2,
                                 (self.scanner.nCrystalRings - 1) / 2 + 1))
        axialCoorPerSeg = []
        r = self.scanner.effDetectorRadiusCm

        for i in range(self.sinogram.nSegments):
            n_planes = self.sinogram.numberOfPlanesPerSeg[i]
            zy       = np.zeros([n_planes, 4])
            for j in range(n_planes):
                ii, jj    = self._col2ij(Segments[i][j], self.scanner.nCrystalRings)
                zy[j, 0]  = np.mean(z_axis[ii])
                zy[j, 1]  = np.mean(z_axis[jj])
                zy[j, 2]  =  r
                zy[j, 3]  = -r
            axialCoorPerSeg.append(zy)
        return axialCoorPerSeg, z_axis

    def LorsTransaxialCoor(self):
        """Compute transaxial (x, y) coordinates of LOR end-points.

        Returns
        -------
        xy1, xy2 : (nAngularBins/2, nRadialBins, 2) — detector crystal positions
        gaps     : (nRadialBins, nAngularBins/2) bool — virtual crystal flags
        """
        sc = self.scanner
        si = self.sinogram

        startXtal = (sc.nCrystalsPerRing - si.nRadialBins) // 4
        si.startXtal = startXtal

        p         = np.linspace(2 * np.pi, 0, sc.nCrystalsPerRing + 1)
        centerCm  = np.stack([sc.effDetectorRadiusCm * np.cos(p[1:]),
                               sc.effDetectorRadiusCm * np.sin(p[1:])], axis=1)
        isVirtual = np.zeros(sc.nCrystalsPerRing, dtype=bool)
        idx_virt  = np.arange(sc.nPhysCrystalsPerBlock + 1,
                               sc.nCrystalsPerRing + sc.nPhysCrystalsPerBlock + 1,
                               sc.nPhysCrystalsPerBlock + 1)
        isVirtual[idx_virt - 1] = True

        increment = np.zeros([sc.nCrystalsPerRing, 2], dtype="int16")
        increment[0::2, 0] = np.arange(1, sc.nCrystalsPerRing // 2 + 1)
        increment[1::2, 0] = increment[0::2, 0] + 1
        increment[0::2, 1] = np.arange(sc.nCrystalsPerRing // 2 + 1, sc.nCrystalsPerRing + 1)
        increment[1::2, 1] = increment[0::2, 1]

        halfRad = si.nRadialBins // 2 + 1
        R       = np.empty((sc.nCrystalsPerRing, 3), dtype=object)
        V       = np.zeros([halfRad, 2], dtype="int16")

        for ii in range(sc.nCrystalsPerRing):
            s1 = self._rem_p(startXtal + np.arange(0, halfRad, dtype="int16") - increment[ii, 0],
                             sc.nCrystalsPerRing) - 1
            s2 = self._rem_p(startXtal + np.arange(0, halfRad, dtype="int16") - increment[ii, 1],
                             sc.nCrystalsPerRing) - 1
            s2      = s2[::-1]
            V[:, 0] = isVirtual[s1]
            V[:, 1] = isVirtual[s2]
            R[ii, 0] = centerCm[s1, :]
            R[ii, 1] = centerCm[s2, :]
            R[ii, 2] = V.copy()

        # Interleave even / odd radial bins
        xy1  = np.zeros([sc.nCrystalsPerRing // 2, si.nRadialBins, 2])
        xy2  = np.zeros([sc.nCrystalsPerRing // 2, si.nRadialBins, 2])
        gaps = np.zeros([sc.nCrystalsPerRing // 2, si.nRadialBins], dtype="int16")

        for i in range(sc.nCrystalsPerRing // 2):
            idx   = sc.nCrystalsPerRing - (2 * i + 2)
            P1, P2 = R[idx, 0], R[idx, 1]
            xy1[i, 0::2, :] = P1[:-1, :]
            xy1[i, 1::2, :] = (P1[:-1, :] + P1[1:, :]) / 2
            xy2[i, 0::2, :] = P2[:-1, :]
            xy2[i, 1::2, :] = (P2[:-1, :] + P2[1:, :]) / 2
            a = (np.sum(R[idx + 1, 2], axis=1) > 0).reshape(-1, 1)
            b = (np.sum(R[idx,     2], axis=1) > 0).reshape(-1, 1)
            gaps[i, :] = np.concatenate([a, b], axis=1).flatten()[1:-1]

        if si.nMash == 2:
            xy1  = (xy1[0::2] + xy1[1::2]) / 2
            xy2  = (xy2[0::2] + xy2[1::2]) / 2
            gap2 = np.zeros([si.nAngularBins, si.nRadialBins], dtype="int16")
            for i in range(si.nAngularBins):
                gap2[i, :] = np.sum(gaps[2 * i: 2 * i + 2, :], axis=0)
            gaps = gap2

        gaps = np.transpose(gaps)

        # Store angular sampling metric
        cb   = si.nRadialBins // 2 - 1
        lor1 = xy2[0, cb, :] - xy1[0, cb, :]
        lor2 = xy2[1, cb, :] - xy1[1, cb, :]
        cos_theta = np.dot(lor1, lor2) / (np.linalg.norm(lor1) * np.linalg.norm(lor2))
        si.angSamplingDegrees = float(np.arccos(cos_theta) * 180 / np.pi)

        return xy1, xy2, gaps

    def Lors3DEndPointCoor(self, reduce4symmetries: int = 0):
        """Combine axial and transaxial LOR coordinates into 3-D end-point arrays."""
        axialCoorPerSeg, _ = self.LorsAxialCoor()
        xy1, xy2, gaps     = self.LorsTransaxialCoor()
        si  = self.sinogram
        nA  = si.nAngularBins
        nR  = si.nRadialBins
        nP  = si.totalNumberOfSinogramPlanes

        xyz01 = np.zeros([nA, nR, 3, nP], dtype="float32")
        xyz02 = np.zeros([nA, nR, 3, nP], dtype="float32")
        k     = 0
        flag  = True
        centralSeg = si.nSegments // 2

        for j in range(si.nSegments):
            for i in range(si.numberOfPlanesPerSeg[j]):
                z1 = axialCoorPerSeg[j][i, 0] * np.ones([nA, nR])
                z2 = axialCoorPerSeg[j][i, 1] * np.ones([nA, nR])
                if j > centralSeg:
                    z1, z2 = z2, z1
                    if flag:
                        xy1 = -xy1;  xy2 = -xy2;  flag = False
                xyz01[:, :, 0:2, k] = xy1;  xyz01[:, :, 2, k] = z1
                xyz02[:, :, 0:2, k] = xy2;  xyz02[:, :, 2, k] = z2
                k += 1

        cumPlanes  = np.cumsum(si.numberOfPlanesPerSeg)
        planeRange = np.zeros([len(cumPlanes), 2], dtype="int16")
        planeRange[1:, 0] = cumPlanes[:-1];  planeRange[:, 1] = cumPlanes

        o = np.zeros([si.nSegments], dtype="int16")
        o[0::2] = np.arange(centralSeg, si.nSegments)
        o[1::2] = np.arange(centralSeg - 1, -1, -1)

        newCumPlanes  = np.cumsum(si.numberOfPlanesPerSeg[o])
        newPlaneRange = np.zeros([len(newCumPlanes), 2], dtype="int16")
        newPlaneRange[1:, 0] = newCumPlanes[:-1];  newPlaneRange[:, 1] = newCumPlanes

        si.numberOfPlanesPerSeg    = si.numberOfPlanesPerSeg[o]
        si.originalSegmentOrder    = o

        if self.scanner.model_number == 2008:
            S = np.zeros_like(newPlaneRange)
            S[0, :] = newPlaneRange[0, :]
            for i in range(centralSeg):
                S[2 * i + 1, :] = newPlaneRange[2 * i + 2, :]
                S[2 * i + 2, :] = newPlaneRange[2 * i + 1, :]
            newPlaneRange = S
        si.planeRange = newPlaneRange

        xyz1 = np.zeros_like(xyz01);  xyz2 = np.zeros_like(xyz02)
        for i in range(si.nSegments):
            s, e  = planeRange[o[i], 0], planeRange[o[i], 1]
            ns, ne = newPlaneRange[i, 0], newPlaneRange[i, 1]
            xyz1[:, :, :, ns:ne] = xyz01[:, :, :, s:e]
            xyz2[:, :, :, ns:ne] = xyz02[:, :, :, s:e]

        if reduce4symmetries == 1:
            self.calculateAxialSymmetries()
            uniq  = self.sinogram.uniqueAxialPlanes - 1
            xyz1  = xyz1[0:nA // 2, :, :, uniq]
            xyz2  = xyz2[0:nA // 2, :, :, uniq]

        return xyz1, xyz2, newPlaneRange

    def calculateAxialSymmetries(self) -> None:
        """Pre-compute axial translational / mirror symmetries (3-D only)."""
        pr     = self.sinogram.planeRange.copy()
        pr[:, 0] += 1
        l, c   = self.sinogram.span // 2 + 1, self.sinogram.nSegments // 2
        K      = np.zeros([c, l], dtype="int16")
        K[:, 0] = pr[1::2, 0]
        K[:, 1] = K[:, 0] + 1
        for i in range(2, l):
            K[:, i] = K[:, i - 1] + 2
        self.sinogram.uniqueAxialPlanes = np.concatenate([[1], K.flatten()])

        b  = pr.flatten()
        b  = np.reshape(b[2:], (4, c), order="F").T
        n  = self.sinogram.span - 1
        I  = np.zeros([n, 4], dtype="int16")
        x  = np.arange(n)
        P  = []
        for i in range(c):
            a = b[i, :]
            I = np.zeros_like(I)
            I[:, 0] = a[0] + x;  I[:, 1] = a[1] - x
            I[:, 2] = a[2] + x;  I[:, 3] = a[3] - x
            P.append(I.copy())
        P = np.concatenate(P, axis=0)

        symID = np.zeros(P.shape[0], dtype="int16")
        for i in range(1, len(self.sinogram.uniqueAxialPlanes)):
            j = P[:, 0] == self.sinogram.uniqueAxialPlanes[i]
            symID[j] = i + 1
        zeros = np.nonzero(symID == 0)[0]
        symID[zeros] = symID[zeros - 1]

        nTotal  = self.sinogram.totalNumberOfSinogramPlanes
        Ax      = np.zeros(nTotal, dtype="int16")
        mirror  = np.ones(nTotal, dtype="int16")
        Ax[:self.sinogram.numberOfPlanesPerSeg[0]] = 1
        idx     = n * np.arange(1, c + 1)

        for i in range(len(symID)):
            if np.any((i + 1) == idx):
                ii       = np.concatenate([np.arange(P[i, 0], P[i, 1] + 1),
                                            np.arange(P[i, 2], P[i, 3] + 1)])
                Ax[ii - 1] = symID[i]
                mirror[np.arange(P[i, 2], P[i, 3]) - 1] = -1
            Ax[P[i, :] - 1]      = symID[i]
            mirror[P[i, 2:4] - 1] = -1

        offset = np.zeros(nTotal, dtype="int16")
        for i in range(nTotal):
            if mirror[i] == 1:
                offset[i] = (i + 1) - self.sinogram.uniqueAxialPlanes[Ax[i] - 1]
            else:
                j       = np.nonzero(symID == Ax[i])[0][0]
                xv      = P[j, 0:3]
                offset[i] = (self.image.matrixSize[2] - 1) - (xv[1] - xv[0]) + ((i + 1) - xv[2])

        pmt = np.zeros([nTotal, 3], dtype="int16")
        pmt[:, 0] = Ax;  pmt[:, 1] = mirror;  pmt[:, 2] = offset
        self.sinogram.planeMirrorTranslation = pmt

    # ── System matrix ──────────────────────────────────────────────────────────

    def calculateSystemMatrixPerPlane(self, xyz1: np.ndarray, xyz2: np.ndarray,
                                       I: int, reconFovRadious: float = None):
        """Trace all LORs for sinogram plane *I* using the Siddon algorithm.

        Returns
        -------
        sMatrix   : (nAngularBins//2, nRadialBins) object array of (N, 4) int16 entries
                    Columns: [voxel_x, voxel_y, voxel_z, intersection_length*1e4]
        tofMatrix : matching TOF weight array, or 0 if TOF is disabled
        """
        if reconFovRadious is None:
            reconFovRadious = self.scanner.transaxialFovCm / 2.5

        img = self.image
        si  = self.sinogram
        Nx, Ny, Nz = img.matrixSize[0] + 1, img.matrixSize[1] + 1, img.matrixSize[2] + 1
        dx, dy, dz = img.voxelSizeCm
        bx, by, bz = -(Nx - 1) * dx / 2, -(Ny - 1) * dy / 2, -(Nz - 1) * dz / 2
        vCy = dx * np.arange(-(Nx - 2) / 2, (Nx - 2) / 2 + 1)
        vCx = dy * np.arange(-(Ny - 2) / 2, (Ny - 2) / 2 + 1)
        vCz = dz * np.arange(-(Nz - 2) / 2, (Nz - 2) / 2 + 1)
        WEAK_THRESHOLD = 50

        sMatrix   = np.zeros((si.nAngularBins // 2, si.nRadialBins), dtype=object)
        tofMatrix = 0

        if self.scanner.isTof:
            from scipy.stats import norm as _norm
            tofBounds = np.linspace(-self.scanner.coinciWindowWidthNsec / 2,
                                     self.scanner.coinciWindowWidthNsec / 2,
                                     si.nTofBins + 1)
            sigma_tof = self.scanner.tofResolutionNsec / np.sqrt(np.log(256))
            tofMatrix = np.zeros((si.nAngularBins // 2, si.nRadialBins), dtype=object)

        def _param_intersect(p1, p2, amin, axmin, amax, axmax, t, b, d, N):
            """Siddon parametric intersection along one axis."""
            if p1 < p2:
                imin = 1 if amin == axmin else int(np.ceil(((p1 + amin * t) - b) / d))
                imax = N - 1 if amax == axmax else int(np.floor(((p1 + amax * t) - b) / d))
                return (b + np.arange(imin, imax + 1) * d - p1) / t
            else:
                imax = N - 2 if amin == axmin else int(np.floor(((p1 + amin * t) - b) / d))
                imin = 0 if amax == axmax else int(np.ceil(((p1 + amax * t) - b) / d))
                return (b + np.arange(imax, imin - 1, -1) * d - p1) / t

        for ang in range(si.nAngularBins // 2):
            for rad in range(si.nRadialBins):
                p1x, p1y, p1z = xyz1[ang, rad, 0, I], xyz1[ang, rad, 1, I], xyz1[ang, rad, 2, I]
                p2x, p2y, p2z = xyz2[ang, rad, 0, I], xyz2[ang, rad, 1, I], xyz2[ang, rad, 2, I]

                tx = p2x - p1x or (p2x + 1e-2 - p1x)
                ty = p2y - p1y or (p2y + 1e-2 - p1y)
                tz = p2z - p1z or (p2z + 1e-2 - p1z)
                # (correct handling of zero denominators is in the original loop below)

                # re-do properly with the original zero-check pattern
                if p2x - p1x == 0: p2x += 1e-2
                if p2y - p1y == 0: p2y += 1e-2
                if p2z - p1z == 0: p1z += 1e-2
                tx = p2x - p1x;  ty = p2y - p1y;  tz = p2z - p1z

                ax_arr = (bx + np.array([0, Nx - 1]) * dx - p1x) / tx
                ay_arr = (by + np.array([0, Ny - 1]) * dy - p1y) / ty
                az_arr = (bz + np.array([0, Nz - 1]) * dz - p1z) / tz
                axmin, axmax = ax_arr.min(), ax_arr.max()
                aymin, aymax = ay_arr.min(), ay_arr.max()
                azmin, azmax = az_arr.min(), az_arr.max()

                amin = max(0.0, axmin, aymin, azmin)
                amax = min(1.0, axmax, aymax, azmax)

                if amin >= amax:
                    continue

                ax_ = _param_intersect(p1x, p2x, amin, axmin, amax, axmax, tx, bx, dx, Nx)
                ay_ = _param_intersect(p1y, p2y, amin, aymin, amax, aymax, ty, by, dy, Ny)
                az_ = _param_intersect(p1z, p2z, amin, azmin, amax, azmax, tz, bz, dz, Nz)
                a   = np.unique(np.concatenate([[amin], ax_, ay_, az_, [amax]]))
                k   = np.arange(len(a) - 1)
                am  = (a[k + 1] + a[k]) / 2

                im = np.floor(((p1x + am * tx) - bx) / dx).astype("int32")
                jm = np.floor(((p1y + am * ty) - by) / dy).astype("int32")
                km = np.floor(((p1z + am * tz) - bz) / dz).astype("int32")
                LL = ((a[k + 1] - a[k]) * np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
                      * 1e4 / dx).astype("int16")

                M = np.stack([im, jm, km, LL], axis=1)
                M = M[M[:, 3] > WEAK_THRESHOLD, :]
                if reconFovRadious != 0:
                    in_fov = (vCy[M[:, 0]] ** 2 + vCx[M[:, 1]] ** 2) < reconFovRadious ** 2
                    M = M[in_fov, :]

                if M.size:
                    sMatrix[ang, rad] = M
                    if self.scanner.isTof:
                        Vc = np.stack([vCy[M[:, 0]], vCx[M[:, 1]], vCz[M[:, 2]]], axis=1)
                        ep1 = np.tile([p1x, p1y, p1z], (Vc.shape[0], 1))
                        ep2 = np.tile([p2x, p2y, p2z], (Vc.shape[0], 1))
                        dL  = np.linalg.norm(ep2 - Vc, axis=1) - np.linalg.norm(ep1 - Vc, axis=1)
                        dT  = dL / 30 - self.scanner.tofOffsetNsec
                        W   = np.zeros([Vc.shape[0], si.nTofBins])
                        for q in range(Vc.shape[0]):
                            cdf      = _norm.cdf(tofBounds, dT[q], sigma_tof)
                            W[q, :]  = cdf[1:] - cdf[:-1]
                        tofMatrix[ang, rad] = (W * 1e4).astype("int16")

        return sMatrix, tofMatrix

    def buildSystemMatrixUsingSymmetries(self, save_dir: str = None,
                                          reconFovRadious: float = None,
                                          is3d: bool = False,
                                          ncores: int = 1) -> None:
        """Compute and save system matrix files (geoMatrix-N.npy)."""
        if save_dir is None:
            save_dir = os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Building system matrix → %s", save_dir)
        save_dir = save_dir + self.engine.bar

        if reconFovRadious is None:
            reconFovRadious = self.scanner.transaxialFovCm / 2.5

        params = {"reconFovRadious": reconFovRadious,
                  "radialBinCropfactor": self.sinogram.radialBinCropfactor}
        np.save(save_dir + "parameters.npy", params)

        xyz1, xyz2, _ = self.Lors3DEndPointCoor(1)
        N = len(self.sinogram.uniqueAxialPlanes) if is3d else 1

        if ncores == 1:
            for i in range(N):
                logger.info("  Plane %d / %d ...", i + 1, N)
                geo, tof = self.calculateSystemMatrixPerPlane(xyz1, xyz2, i, reconFovRadious)
                np.save(f"{save_dir}geoMatrix-{i}.npy", geo)
                if self.scanner.isTof:
                    np.save(f"{save_dir}tofMatrix-{i}.npy", tof)
        else:
            logger.warning("Multi-core system-matrix build not yet implemented; running single-core.")
            self.buildSystemMatrixUsingSymmetries(save_dir.rstrip(self.engine.bar),
                                                   reconFovRadious, is3d, ncores=1)

    def loadSystemMatrix(self, save_dir: str, is3d: bool = False,
                         tof: bool = True, reconFovRadious: float = None) -> None:
        """Load pre-computed system matrix files from *save_dir*.

        If the files do not exist, they are computed automatically (slow).
        """
        self.is3d = is3d
        path      = save_dir + self.engine.bar
        tic       = time.time()

        needs_build = (
            (not is3d and not os.path.isfile(path + "geoMatrix-0.npy")) or
            (is3d     and not os.path.isfile(path + "geoMatrix-1.npy"))
        )
        if needs_build:
            logger.info("System matrix files not found — computing now (this may take a while) ...")
            self.buildSystemMatrixUsingSymmetries(save_dir, is3d=is3d,
                                                   reconFovRadious=reconFovRadious)

        self.geoMatrix = []
        param = np.load(path + "parameters.npy", allow_pickle=True).item()
        self.image.reconFovRadious = param["reconFovRadious"]

        if self.is3d:
            if not hasattr(self.sinogram, "uniqueAxialPlanes"):
                self.Lors3DEndPointCoor(1)
            N = len(self.sinogram.uniqueAxialPlanes)
            if tof and self.scanner.isTof:
                self.tofMatrix = []
            for i in range(N):
                self.geoMatrix.append(np.load(f"{path}geoMatrix-{i}.npy"))
                if tof and self.scanner.isTof:
                    self.tofMatrix.append(np.load(f"{path}tofMatrix-{i}.npy"))
        else:
            self.buildMichelogram()
            if self.sinogram.radialBinCropfactor != param["radialBinCropfactor"]:
                raise ValueError(
                    f"radialBinCropfactor mismatch: current={self.sinogram.radialBinCropfactor!r}, "
                    f"stored={param['radialBinCropfactor']!r}. "
                    f"Choose a different save_dir or delete {path}geoMatrix-0.npy to recompute."
                )
            self.geoMatrix.append(
                np.load(path + "geoMatrix-0.npy", allow_pickle=True)
            )
            if tof and self.scanner.isTof:
                self.tofMatrix = [np.load(path + "tofMatrix-0.npy", allow_pickle=True)]

        logger.info("System matrix loaded in %.1f s.", time.time() - tic)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _zero_pad(self, y):
        """Zero-pad a list of 1-D arrays to the same length (Michelogram helper)."""
        max_len     = max(len(y[i]) for i in range(len(y)))
        out         = np.zeros([len(y), max_len], dtype="int16")
        for i in range(len(y)):
            offset  = (max_len - len(y[i])) // 2
            if offset == 0:
                out[i, :] = y[i]
            else:
                out[i, offset: -offset] = y[i]
        return out, max_len

    def _zero_trim(self, y):
        """Strip trailing zeros from each column of a padded 2-D array."""
        return [y[:, i][np.nonzero(y[:, i])] for i in range(y.shape[1])]

    def _col2ij(self, m, n: int):
        """Convert column-major linear index to (row, col) subscripts."""
        if np.max(m) > n ** 2:
            raise ValueError("Index exceeds matrix size.")
        j = np.ceil(m / n) - 1
        i = m - j * n - 1
        return i.astype(int), j.astype(int)

    def _rem_p(self, x: np.ndarray, nx: int) -> np.ndarray:
        """Wrap indices into [1, nx] (modular arithmetic, 1-based)."""
        for i in range(len(x)):
            while x[i] < nx:
                x[i] += nx
            while x[i] > nx:
                x[i] -= nx
        return x

    # ── Plot helpers ───────────────────────────────────────────────────────────

    def plotLorsAxialCoor(self, plotSeparateSegmentsToo: bool = False) -> None:
        from matplotlib import pyplot as plt
        axialCoorPerSeg, z_axis = self.LorsAxialCoor()
        r = self.scanner.effDetectorRadiusCm
        plt.figure()
        for j in range(self.sinogram.nSegments):
            for i in range(len(axialCoorPerSeg[j])):
                plt.plot(axialCoorPerSeg[j][i, 0:2], axialCoorPerSeg[j][i, 2:4],
                         color="green", linestyle="solid")
        plt.plot(z_axis, r * np.ones_like(z_axis), "bs", fillstyle="none", markeredgewidth=2)
        plt.plot(z_axis, -r * np.ones_like(z_axis), "bs", fillstyle="none", markeredgewidth=2)
        plt.xlabel("Axial Distance (cm)", fontsize=18)
        plt.ylabel("Radial Distance (cm)", fontsize=18)
        plt.title("All Segments", fontsize=18)

    def plotLorsTransaxialCoor(self) -> None:
        from matplotlib import pyplot as plt
        xy1, xy2, gaps = self.LorsTransaxialCoor()
        lim = self.scanner.transaxialFovCm * 3 / 4
        for i in range(self.sinogram.nAngularBins // 4):
            plt.clf()
            plt.plot(np.array([xy1[i, :, 0], xy2[i, :, 0]]),
                     np.array([xy1[i, :, 1], xy2[i, :, 1]]),
                     c="green", ls="-", lw=0.5)
            idx = (gaps[:, i] > 1) if self.sinogram.nMash == 2 else (gaps[:, i] > 0)
            plt.plot(np.array([xy1[i, idx, 0], xy2[i, idx, 0]]),
                     np.array([xy1[i, idx, 1], xy2[i, idx, 1]]),
                     c="blue", ls="-", lw=0.75)
            plt.axis("square")
            plt.xlim(-lim, lim);  plt.ylim(-lim, lim)
            plt.title(f"Angle: {i + 1}", fontsize=15)
            plt.show();  plt.pause(0.1)
