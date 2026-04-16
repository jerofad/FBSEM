"""
3D PET geometry — extends geometry.BuildGeometry_v4 with full 3D reconstruction.

All scanner geometry, 2D sinogram/image utilities, and 2D OSEM/MAPEM algorithms
are inherited from the parent class.  This module adds:
  - Python-based 3D forward/back-projection (forwardProject3D, MLEM3D_python, OSEM3D_python)
  - APIRL-based GPU 3D projections (forwardProjectBatch3D, backProjectBatch3D, OSEM3D, MAPEM3D)
  - E7-tools sinogram I/O helpers (get_e7sino, iSSRB, read_sino)

Usage:
    from geometry.geometry3d.BuildGeometry_v4 import BuildGeometry_v4
    PET = BuildGeometry_v4('mmr', 0.5)
    PET.setApirlMmrEngine(binPath='/path/to/apirl/bin', temPath='/tmp/pet')
"""

import numpy as np
import os
import subprocess
from scipy import ndimage
import multiprocessing as mp

np.seterr(divide='ignore')

from geometry.BuildGeometry_v4 import BuildGeometry_v4 as _BuildGeometry2D, dotstruct  # noqa: F401 (re-exported)


class BuildGeometry_v4(_BuildGeometry2D):
    """2D + 3D PET geometry.  Inherits all 2D methods; adds 3D reconstruction."""

    # ------------------------------------------------------------------
    # Backward-compatibility alias: old code called OSMAPEM2D_DePierro
    # ------------------------------------------------------------------
    def OSMAPEM2D_DePierro(self, prompts, img=None, RS=None, niter=100, nsubs=1,
                            AN=None, tof=False, psf=0, beta=1, prior=None, prior_weights=1):
        """Alias for MAPEM2D (kept for backward compatibility)."""
        return self.MAPEM2D(prompts, img, RS, niter, nsubs, AN, tof, psf, beta, prior, prior_weights)

    # ------------------------------------------------------------------
    # Python-based 3D reconstruction (slow; reference implementations)
    # ------------------------------------------------------------------
    def forwardProject3D(self, img3d, tof=False, psf=0):
        import time
        if tof and not self.scanner.isTof:
           raise ValueError("The scanner is not TOF")
        nUniqueAxialPlanes = len(self.sinogram.uniqueAxialPlanes)
        allPlanes = []
        for i in range(len(self.sinogram.uniqueAxialPlanes)):
            allPlanes.append(np.nonzero(self.sinogram.planeMirrorTranslation[:,0] == i+1)[0])

        img3d = self.gaussFilter(img3d.flatten('F'), psf)
        dims = [self.sinogram.nRadialBins, self.sinogram.nAngularBins, self.sinogram.totalNumberOfSinogramPlanes]
        if tof: dims.append(self.sinogram.nTofBins)
        y = np.zeros(dims, dtype='float')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins // 2
        planeMirrorTranslation = self.sinogram.planeMirrorTranslation
        prod = lambda x, y: x.reshape(-1,1).dot(y.reshape(1,-1)).T.astype('int32')
        tic = time.time()
        for i in range(self.sinogram.nAngularBins // 2):
            for j in range(self.sinogram.nRadialBins):
                for p in range(nUniqueAxialPlanes):
                    M0 = self.geoMatrix[p][i, j]
                    if not np.isscalar(M0):
                        M = M0[:, 0:3].astype('int32')
                        G = M0[:, 3] / 1e4
                        H = planeMirrorTranslation[allPlanes[p], :]
                        idxAxial = matrixSize[0]*matrixSize[1]*(prod(H[:,1], M[:,2]) + H[:,2])
                        idx1 = (M[:,0] + M[:,1]*matrixSize[0]).reshape(-1,1) + idxAxial
                        idx2 = (M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])).reshape(-1,1) + idxAxial
                        y[j, i, allPlanes[p]] = G.dot(img3d[idx1])
                        y[j, i+q, allPlanes[p]] = G.dot(img3d[idx2])
        print('forward-projected in: {} sec.'.format(time.time()-tic))
        return y

    def MLEM3D_python(self, prompts, img=None, RS=None, niter=100, AN=None, tof=False, psf=0):
        import time
        if tof and not self.scanner.isTof:
               raise ValueError("The scanner is not TOF")
        if img is None:
            img = np.ones(self.image.matrixSize, dtype='float')
        img = img.flatten('F')
        sensImage = np.zeros_like(img)
        if np.ndim(prompts) != 4:
            tof = False
        if RS is None:
            RS = 0*prompts
        if AN is None:
            AN = np.ones([self.sinogram.nRadialBins, self.sinogram.nAngularBins,
                          self.sinogram.totalNumberOfSinogramPlanes], dtype='float')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins // 2
        Flag = True
        nUniqueAxialPlanes = len(self.sinogram.uniqueAxialPlanes)
        allPlanes = []
        for i in range(len(self.sinogram.uniqueAxialPlanes)):
            allPlanes.append(np.nonzero(self.sinogram.planeMirrorTranslation[:,0] == i+1)[0])
        planeMirrorTranslation = self.sinogram.planeMirrorTranslation
        prod = lambda x, y: x.reshape(-1,1).dot(y.reshape(1,-1)).T.astype('int32')
        tic = time.time()
        for n in range(niter):
            if np.any(psf != 0):
                imgOld = self.gaussFilter(img, psf, True)
            else:
                imgOld = img
            backProjImage = 0*img
            for i in range(self.sinogram.nAngularBins // 2):
                for j in range(self.sinogram.nRadialBins):
                    for p in range(nUniqueAxialPlanes):
                        M0 = self.geoMatrix[p][i, j]
                        if not np.isscalar(M0):
                            M = M0[:, 0:3].astype('int32')
                            G = M0[:, 3] / 1e4
                            H = planeMirrorTranslation[allPlanes[p], :]
                            idxAxial = matrixSize[0]*matrixSize[1]*(prod(H[:,1], M[:,2]) + H[:,2])
                            idx1 = (M[:,0] + M[:,1]*matrixSize[0]).reshape(-1,1) + idxAxial
                            idx2 = (M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])).reshape(-1,1) + idxAxial
                            if tof:
                                W = self.tofMatrix[p][i, j] / 1e4
                                f1 = AN[j,i,allPlanes[p]].reshape(-1,1) * (prompts[j,i,allPlanes[p],:] /
                                     ((G.reshape(-1,1)*imgOld[idx1]).T.dot(W) + RS[j,i,allPlanes[p],:] + 1e-5))
                                f2 = AN[j,i+q,allPlanes[p]].reshape(-1,1) * (prompts[j,i+q,allPlanes[p],:] /
                                     ((G.reshape(-1,1)*imgOld[idx2]).T.dot(W) + RS[j,i+q,allPlanes[p],:] + 1e-5))
                                backProjImage[idx1] += G.reshape(-1,1)*W.dot(f1.T)
                                backProjImage[idx2] += G.reshape(-1,1)*W.dot(f2.T)
                            else:
                                f1 = AN[j,i,allPlanes[p]] * (prompts[j,i,allPlanes[p]] /
                                     (AN[j,i,allPlanes[p]]*G.dot(imgOld[idx1]) + RS[j,i,allPlanes[p]] + 1e-5))
                                f2 = AN[j,i+q,allPlanes[p]] * (prompts[j,i+q,allPlanes[p]] /
                                     (AN[j,i+q,allPlanes[p]]*G.dot(imgOld[idx2]) + RS[j,i+q,allPlanes[p]] + 1e-5))
                                backProjImage[idx1] += G.reshape(-1,1).dot(f1.reshape(1,-1))
                                backProjImage[idx2] += G.reshape(-1,1).dot(f2.reshape(1,-1))
                            if Flag:
                                if tof:
                                    GW = G*np.sum(W, axis=1)
                                    sensImage[idx1] += GW.reshape(-1,1).dot(AN[j,i,allPlanes[p]].reshape(1,-1))
                                    sensImage[idx2] += GW.reshape(-1,1).dot(AN[j,i+q,allPlanes[p]].reshape(1,-1))
                                else:
                                    sensImage[idx1] += G.reshape(-1,1).dot(AN[j,i,allPlanes[p]].reshape(1,-1))
                                    sensImage[idx2] += G.reshape(-1,1).dot(AN[j,i+q,allPlanes[p]].reshape(1,-1))
            if np.any(psf != 0) and Flag:
                sensImage = self.gaussFilter(sensImage, psf, True)
            Flag = False
            img = imgOld * backProjImage / (sensImage + 1e-5)
        print('forward-projected in: {} sec.'.format((time.time()-tic)/60))
        return img.reshape(matrixSize, order='F')

    def OSEM3D_python(self, prompts, img=None, RS=None, niter=100, nsubs=1, AN=None, psf=0):
        import time
        [numAng, subSize] = self.angular_subsets(nsubs)
        if img is None:
            img = np.ones(self.image.matrixSize, dtype='float')
        img = img.flatten('F')
        sensImage = np.zeros_like(img)
        sensImageSubs = np.zeros((np.prod(self.image.matrixSize), nsubs), dtype='float')
        if RS is None:
            RS = 0*prompts
        if AN is None:
            AN = np.ones([self.sinogram.nRadialBins, self.sinogram.nAngularBins,
                          self.sinogram.totalNumberOfSinogramPlanes], dtype='float')
        matrixSize = self.image.matrixSize
        q = self.sinogram.nAngularBins // 2
        Flag = True
        nUniqueAxialPlanes = len(self.sinogram.uniqueAxialPlanes)
        allPlanes = []
        for i in range(len(self.sinogram.uniqueAxialPlanes)):
            allPlanes.append(np.nonzero(self.sinogram.planeMirrorTranslation[:,0] == i+1)[0])
        planeMirrorTranslation = self.sinogram.planeMirrorTranslation
        prod = lambda x, y: x.reshape(-1,1).dot(y.reshape(1,-1)).T.astype('int32')
        tic = time.time()
        for n in range(niter):
             for sub in range(nsubs):
                 imgOld = self.gaussFilter(img, psf, True)
                 backProjImage = 0*img
                 for ii in range(subSize // 2):
                     i = numAng[ii, sub]
                     for j in range(self.sinogram.nRadialBins):
                         for p in range(nUniqueAxialPlanes):
                             M0 = self.geoMatrix[p][i, j]
                             if not np.isscalar(M0):
                                 M = M0[:, 0:3].astype('int32')
                                 G = M0[:, 3] / 1e4
                                 H = planeMirrorTranslation[allPlanes[p], :]
                                 idxAxial = matrixSize[0]*matrixSize[1]*(prod(H[:,1], M[:,2]) + H[:,2])
                                 idx1 = (M[:,0] + M[:,1]*matrixSize[0]).reshape(-1,1) + idxAxial
                                 idx2 = (M[:,1] + matrixSize[0]*(matrixSize[0]-1-M[:,0])).reshape(-1,1) + idxAxial
                                 an = AN[j, i, allPlanes[p]]
                                 f = an * (prompts[j,i,allPlanes[p]] /
                                           (an*G.dot(imgOld[idx1]) + RS[j,i,allPlanes[p]] + 1e-5))
                                 backProjImage[idx1] += G.reshape(-1,1).dot(f.reshape(1,-1))
                                 an = AN[j, i+q, allPlanes[p]]
                                 f = an * (prompts[j,i+q,allPlanes[p]] /
                                           (an*G.dot(imgOld[idx2]) + RS[j,i+q,allPlanes[p]] + 1e-5))
                                 backProjImage[idx2] += G.reshape(-1,1).dot(f.reshape(1,-1))
                                 if Flag:
                                      sensImage[idx1] += G.reshape(-1,1).dot(AN[j,i,allPlanes[p]].reshape(1,-1))
                                      sensImage[idx2] += G.reshape(-1,1).dot(AN[j,i+q,allPlanes[p]].reshape(1,-1))
                 if Flag:
                      sensImageSubs[:, sub] = sensImage + 1e-5
                 backProjImage = self.gaussFilter(backProjImage, psf)
                 img = imgOld * backProjImage / (sensImageSubs[:, sub])
             Flag = False
             sensImage = 0
        print('forward-projected in: {} min.'.format((time.time()-tic)/60))
        return img.reshape(matrixSize, order='F')

    # ------------------------------------------------------------------
    # APIRL-based GPU 3D reconstruction
    # ------------------------------------------------------------------
    def createConfigFile(self, flname, input_file, output_data_flname, output_filename,
                          gpu=True, project_mode=True, nsubs=1, subsetIndex=0):
          f = open(flname, "w+")
          if project_mode:
               f.write("Projection Parameters :=\n"
                       "output type := Sinogram3DSiemensMmr\n")
               if gpu:
                    f.write("projector := CuSiddonProjector\n"
                            "projector block size := {256,1,1}\n"
                            "gpu id := 0\n")
               else:
                    f.write("projector := Siddon\n")
               f.write(f"output projection := {output_data_flname}\n")
          else:
               f.write("Backproject Parameters :=\n"
                       "input type := Sinogram3DSiemensMmr\n")
               if gpu:
                    f.write("backprojector := CuSiddonProjector\n"
                            "backprojector block size := {576,1,1}\n"
                            "gpu id := 0\n")
               else:
                    f.write("backprojector := Siddon\n")
               f.write(f"output image := {output_data_flname}\n")
          f.write("siddon number of samples on the detector := 1\n"
                  "siddon number of axial samples on the detector := 1\n")
          if nsubs > 1:
               f.write(f"number of subsets := {nsubs}\n")
               f.write(f"subset index := {subsetIndex}\n")
          f.write(f"input file := {input_file}\n")
          f.write(f"output filename := {output_filename}\n")
          f.close()

    def write_to_apirl(self, flname, data=None, sino_mode=False):
          if data is None:
               dataType = 'short float'
               itemsize = 4
          else:
               if data.dtype.name == 'int32':
                   dataType = 'signed integer'
               elif data.dtype.name == 'int16':
                    dataType = 'signed integer'
               elif data.dtype.name == 'float64':
                    dataType = 'long float'
               elif data.dtype.name == 'float32':
                    dataType = 'short float'
               itemsize = data.itemsize
          f = open(flname+".h33", "w+")
          f.write("!INTERFILE :=\n")
          f.write(f"!name of data file := {flname}.i33\n")
          if sino_mode:
               f.write(f"!number format := {dataType}\n")
               f.write(f"!number of bytes per pixel := {itemsize}\n")
               f.write("imagedata byte order := LITTLEENDIAN\n"
               "number of dimensions := 4\n"
               "matrix axis label [4] := segment\n"
               "!matrix size [4] := 11\n"
               "matrix axis label [2] := view\n"
               "!matrix size [2] := 252\n"
               "matrix axis label [3] := axial coordinate\n"
               "!matrix size [3] := { 127, 115, 115, 93, 93, 71, 71, 49, 49, 27, 27 }\n"
               "matrix axis label [1] := tangential coordinate\n")
               f.write(f"!matrix size [1] := {self.sinogram.nRadialBins}\n")
               f.write("minimum ring difference per segment := {  -5, 6, -16, 17, -27, 28, -38, 39, -49, 50, -60 }\n"
               "maximum ring difference per segment := {  5, 16, -6, 27, -17, 38, -28, 49, -39, 60, -50 }\n"
               "number of rings := 64\n"
               "!END OF INTERFILE :=\n")
          else:
               matrix_size = data.shape
               f.write(f"!total number of images := {matrix_size[2]}\n")
               f.write("!imagedata byte order := LITTLEENDIAN\n")
               f.write(f"!matrix size [1] := {matrix_size[0]}\n")
               f.write(f"!matrix size [2] := {matrix_size[1]}\n")
               f.write(f"!matrix size [3] := {matrix_size[2]}\n")
               f.write("!number format := float\n"
               "!number of bytes per pixel := 4\n"
               "scaling factor (mm/pixel) [1] := 2.086260\n"
               "scaling factor (mm/pixel) [2] := 2.086260\n"
               "scaling factor (mm/pixel) [3] := 2.031250\n"
               "!END OF INTERFILE :=\n")
          f.close()
          f = open(flname+".i33", "wb")
          if data is not None:
                f.write(data.tobytes(order='F'))
          f.close()

    def reserve_temPath(self, batch_size):
          tmpath = self.engine.temPath
          if type(tmpath) != list:
               temPaths = [tmpath]
          else:
               if len(tmpath) >= batch_size:
                    self.engine.temPath = tmpath[0:batch_size]
                    return
               else:
                    tmpath = tmpath[0]
                    temPaths = [tmpath]
          for f in range(batch_size - 1):
                temPath = tmpath + str(f)
                if not os.path.exists(temPath):
                     os.makedirs(temPath)
                temPaths.append(temPath)
          self.engine.temPath = temPaths

    def fwd_subprocess(self, img_b, input_img_flname, out_sino_flname, config_flname, nsubs, subsetIndex, psf):
          sino_shape = self.sinogram.shape
          if np.any(psf != 0):
               img_b = self.gaussFilterBatch(img_b, psf)
          self.write_to_apirl(input_img_flname, img_b.transpose(1, 0, 2))
          subprocess.run([self.engine.binPath + self.engine.bar + 'project', config_flname],
                         stdout=subprocess.PIPE)
          tmp = np.fromfile(out_sino_flname+'.i33', dtype='float32')
          if nsubs == 1:
               sino = tmp.reshape(sino_shape, order='F')
          else:
               subshape = [sino_shape[0], sino_shape[1]//nsubs, sino_shape[2]]
               tmp = tmp.reshape(subshape, order='F')
               sino = np.zeros(sino_shape, dtype='float32')
               sino[:, subsetIndex::nsubs, :] = tmp
          sino[np.isnan(sino)] = 0
          os.remove(out_sino_flname+'.h33')
          os.remove(out_sino_flname+'.i33')
          os.remove(input_img_flname+'.h33')
          os.remove(input_img_flname+'.i33')
          return sino

    def forwardProjectBatch3D(self, img, nsubs=1, subsetIndex=0, psf=0):
          if nsubs > 1:
               self.check_nsubs(nsubs)
          img = img.astype('float32')
          if np.ndim(img) == 3:
               batch_size = 1
               img = img[None, :, :, :]
          else:
               batch_size = img.shape[0]
          sino_shape = self.sinogram.shape
          sinoOut = np.zeros((batch_size, sino_shape[0], sino_shape[1], sino_shape[2]), dtype='float32')
          out_sino_flname = []
          sample_sino_flname = []
          input_img_flname = []
          config_flname = []
          num_temp = batch_size if (self.engine.multiprocess and self.engine.gpu) else 1
          self.reserve_temPath(num_temp)
          for b in range(num_temp):
               out_sino_flname.append(self.engine.temPath[b] + self.engine.bar + 'out_sino')
               sample_sino_flname.append(self.engine.temPath[b] + self.engine.bar + 'sample_sino')
               input_img_flname.append(self.engine.temPath[b] + self.engine.bar + 'input_img')
               config_flname.append(self.engine.temPath[b] + self.engine.bar + 'fwdproj.par')
               if not os.path.exists(sample_sino_flname[b]+'.h33'):
                    self.write_to_apirl(sample_sino_flname[b], sino_mode=True)
               self.createConfigFile(config_flname[b], input_img_flname[b]+'.h33',
                                     sample_sino_flname[b]+'.h33', out_sino_flname[b],
                                     self.engine.gpu, True, nsubs, subsetIndex)
          if num_temp > 1:
               pool = mp.Pool(processes=batch_size)
               sino = pool.starmap(self.fwd_subprocess,
                   [(img[b,:,:,:], input_img_flname[b], out_sino_flname[b],
                     config_flname[b], nsubs, subsetIndex, psf) for b in range(batch_size)])
               pool.close()
               for b in range(batch_size):
                    sinoOut[b, :, :, :] = sino[b]
          else:
               for b in range(batch_size):
                    sinoOut[b, :, :, :] = self.fwd_subprocess(
                        img[b,:,:,:], input_img_flname[0], out_sino_flname[0],
                        config_flname[0], nsubs, subsetIndex, psf)
          if batch_size == 1:
               sinoOut = sinoOut[0, :, :, :]
          return sinoOut

    def bkd_subprocess(self, sino_b, input_sino_flname, out_img_flname, config_flname, nsubs, subsetIndex, psf):
          matrix_size = self.image.matrixSize
          self.write_to_apirl(input_sino_flname, sino_b, True)
          subprocess.run([self.engine.binPath + self.engine.bar + 'backproject', config_flname],
                         stdout=subprocess.PIPE)
          img = np.fromfile(out_img_flname+'.i33', dtype='float32').reshape(matrix_size, order='F').transpose(1, 0, 2)
          img[np.isnan(img)] = 0
          if np.any(psf != 0):
               img = self.gaussFilterBatch(img, psf)
          os.remove(out_img_flname+'.h33')
          os.remove(out_img_flname+'.i33')
          os.remove(input_sino_flname+'.h33')
          os.remove(input_sino_flname+'.i33')
          return img * self.mask_fov()

    def backProjectBatch3D(self, sino, nsubs=1, subsetIndex=0, psf=0):
          sino = sino.astype('float32')
          if np.ndim(sino) == 3:
               batch_size = 1
               sino = sino[None, :, :, :]
          else:
               batch_size = sino.shape[0]
          matrix_size = self.image.matrixSize
          imgOut = np.zeros((batch_size, matrix_size[0], matrix_size[1], matrix_size[2]), dtype='float32')
          out_img_flname = []
          sample_img_flname = []
          input_sino_flname = []
          config_flname = []
          num_temp = batch_size if (self.engine.multiprocess and self.engine.gpu) else 1
          self.reserve_temPath(num_temp)
          for b in range(num_temp):
               out_img_flname.append(self.engine.temPath[b] + self.engine.bar + 'out_img')
               sample_img_flname.append(self.engine.temPath[b] + self.engine.bar + 'sample_img')
               input_sino_flname.append(self.engine.temPath[b] + self.engine.bar + 'input_sino')
               config_flname.append(self.engine.temPath[b] + self.engine.bar + 'backproj.par')
               self.createConfigFile(config_flname[b], input_sino_flname[b]+'.h33',
                                     sample_img_flname[b]+'.h33', out_img_flname[b],
                                     self.engine.gpu, False, nsubs, subsetIndex)
               if not os.path.exists(sample_img_flname[b]+'.h33'):
                    self.write_to_apirl(sample_img_flname[b], np.zeros(matrix_size, dtype='float32'))
          if num_temp > 1:
               pool = mp.Pool(processes=batch_size)
               img = pool.starmap(self.bkd_subprocess,
                   [(sino[b,:,:,:], input_sino_flname[b], out_img_flname[b],
                     config_flname[b], nsubs, subsetIndex, psf) for b in range(batch_size)])
               pool.close()
               for b in range(batch_size):
                    imgOut[b, :, :, :] = img[b]
          else:
               for b in range(batch_size):
                    imgOut[b, :, :, :] = self.bkd_subprocess(
                        sino[b,:,:,:], input_sino_flname[0], out_img_flname[0],
                        config_flname[0], nsubs, subsetIndex, psf)
          if batch_size == 1:
               imgOut = imgOut[0, :, :, :]
          return imgOut

    def iSensImageBatch3D(self, AN=None, nsubs=1, psf=0):
          if AN is None:
               AN = np.ones(self.sinogram.shape, dtype='float32')
          if np.ndim(AN) == 3:
               batch_size = 1
               AN = AN[None, :, :, :]
          else:
               batch_size = AN.shape[0]
          sensImgOut = np.zeros((batch_size, nsubs, self.image.matrixSize[0],
                                 self.image.matrixSize[1], self.image.matrixSize[2]), dtype='float32')
          for i in range(nsubs):
               sensImgOut[:, i, :, :, :] = self.backProjectBatch3D(AN, nsubs=nsubs, subsetIndex=i, psf=psf)
          if batch_size == 1:
               sensImgOut = sensImgOut[0, :, :, :, :]
          sensImgOut = 1 / sensImgOut
          sensImgOut[np.isinf(sensImgOut)] = 0
          return sensImgOut

    def forwardDivideBackwardBatch3D(self, img, prompts, RS, AN, nsubs, subsetIndex, psf):
         if RS is None: RS = 0
         y = self.forwardProjectBatch3D(img, nsubs=nsubs, subsetIndex=subsetIndex, psf=psf) + RS + 1e-4
         out = self.backProjectBatch3D(prompts / y, nsubs=nsubs, subsetIndex=subsetIndex, psf=psf)
         out[np.isinf(out)] = 0
         return out

    def OSEM3D(self, prompts, AN=None, RS=None, iSensImg=None, img=None,
               niter=1, nsubs=1, psf=0, display=False):
          if np.ndim(prompts) == 3:
               batch_size = 1
          else:
               batch_size = prompts.shape[0]
          if AN is None:
               sino_shape = ([batch_size] + list(self.sinogram.shape)) if batch_size > 1 else self.sinogram.shape
               AN = np.ones(sino_shape, dtype='float32')
          if iSensImg is None:
               iSensImg = self.iSensImageBatch3D(AN, nsubs, psf)
          if img is None:
               matrix_size = ([batch_size] + list(self.image.matrixSize)) if batch_size > 1 else self.image.matrixSize
               img = np.ones(matrix_size, dtype='float32')
          for n in range(niter):
               if display: print(f"iter: {n}")
               for m in range(nsubs):
                   iSenImg_m = iSensImg[m, :, :, :] if batch_size == 1 else iSensImg[:, m, :, :, :]
                   img = img * self.forwardDivideBackwardBatch3D(img, prompts, RS, AN, nsubs, m, psf) * iSenImg_m
          return img

    def forwardDivideSubtractBackwardBatch3D(self, img, prompts, RS, AN, nsubs, subsetIndex, psf):
         y = self.forwardProjectBatch3D(img, nsubs=nsubs, subsetIndex=subsetIndex, psf=psf) + RS + 1e-4
         out = self.backProjectBatch3D((prompts / y) - 1.0, nsubs=nsubs, subsetIndex=subsetIndex, psf=psf)
         out[np.isinf(out)] = 0
         return out

    def MAPEM3D(self, prompts, AN=None, RS=0, iSensImg=None, img=None,
                niter=1, nsubs=1, psf=0, display=True, beta=1,
                prior_object=None, neighborhood_size=3, weight_type='bowsher',
                weights=None, prior_img=None, bowsher_b=20, gaussian_sigma=0.2):
          """3D MAP-EM using a weighted quadratic prior and De Pierro's convexity lemma."""
          if display: import matplotlib.pyplot as plt
          import time
          if prior_object is None:
               from geometry.Prior import Prior
               prior = Prior(self.image.matrixSize, neighborhood_size)
          else:
               prior = prior_object
          if weights is None:
               if prior_img is None:
                    weights = 1
               else:
                    prior_img_s = self.gaussFilterBatch(prior_img / prior_img.max(), 0.25)
                    if weight_type.lower() == 'bowsher':
                         weights = prior.BowshserWeights(prior_img_s, bowsher_b)
                    else:
                         weights = prior.gaussianWeights(prior_img_s, gaussian_sigma)
          W = (prior.Wd * weights).astype('float32')
          wj = prior.imCropUndo((W.sum(axis=1)).reshape(prior.imageSizeCrop, order='F'))
          wj_ = 1 / wj
          wj_[np.isinf(wj_)] = 0
          if np.ndim(prompts) == 3:
               batch_size = 1
          else:
               batch_size = prompts.shape[0]
          if AN is None:
               sino_shape = ([batch_size] + list(self.sinogram.shape)) if batch_size > 1 else self.sinogram.shape
               AN = np.ones(sino_shape, dtype='float32')
          if iSensImg is None:
               iSensImg = self.iSensImageBatch3D(AN, nsubs, psf)
          if img is None:
               matrix_size = ([batch_size] + list(self.image.matrixSize)) if batch_size > 1 else self.image.matrixSize
               img = np.ones(matrix_size, dtype='float32')
          if np.isscalar(beta):
               beta = np.array([beta], dtype='float32')
               if batch_size > 1:
                    beta = beta[0] * np.ones(batch_size, dtype='float32')
          tic = time.time()
          for n in range(niter):
               for m in range(nsubs):
                   if batch_size == 1:
                        gamma = beta * wj
                        iS = iSensImg[m, :, :, :]
                        img_em = img * self.forwardDivideBackwardBatch3D(img, prompts, RS, AN, nsubs, m, psf) * iS
                        img_reg = img - 0.5*wj_ * prior.GradT(W * prior.Grad(img))
                        img = 2*img_em / ((1 - gamma*iS*img_reg) +
                                          np.sqrt((1 - gamma*iS*img_reg)**2 + 4*gamma*iS*img_em))
                   else:
                        iS = iSensImg[:, m, :, :, :]
                        img_em = img * self.forwardDivideBackwardBatch3D(img, prompts, RS, AN, nsubs, m, psf) * iS
                        for b in range(batch_size):
                             gamma = beta[b] * wj
                             img_reg = img[b,:,:,:] - 0.5*wj_ * prior.GradT(W * prior.Grad(img[b,:,:,:]))
                             img[b,:,:,:] = 2*img_em[b,:,:,:] / (
                                 (1 - gamma*iS[b,:,:,:]*img_reg) +
                                 np.sqrt((1 - gamma*iS[b,:,:,:]*img_reg)**2 + 4*gamma*iS[b,:,:,:]*img_em[b,:,:,:]))
                   if display:
                        plt.imshow((self.crop_img(img, 0.4))[:, :, 50], cmap='gist_heat')
                        plt.pause(0.1)
                        print(f"iter: {n}, sub: {m}")
          print(f'Done in {(time.time()-tic)/60:.3f} min.')
          return img

    def removeSampleFiles(self):
          temPath = self.engine.temPath if type(self.engine.temPath) == str else self.engine.temPath[0]
          dirname = os.path.dirname(temPath)
          folders = os.listdir(dirname)
          for f in folders:
               fn = dirname + self.engine.bar + f + self.engine.bar
               if os.path.exists(fn+'sample_img.h33'):  os.remove(fn+'sample_img.h33')
               if os.path.exists(fn+'sample_img.i33'):  os.remove(fn+'sample_img.i33')
               if os.path.exists(fn+'sample_sino.h33'): os.remove(fn+'sample_sino.h33')
               if os.path.exists(fn+'sample_sino.i33'): os.remove(fn+'sample_sino.i33')

    # ------------------------------------------------------------------
    # E7-tools sinogram I/O helpers
    # ------------------------------------------------------------------
    def get_gaps(self):
          if self.gaps is None:
               _, _, gaps = self.LorsTransaxialCoor()
               gaps2d = ~gaps.astype('bool')
               gaps = np.zeros(self.sinogram.shape, dtype='bool')
               for i in range(gaps.shape[2]):
                    gaps[:, :, i] = gaps2d
               self.gaps = gaps
          return self.gaps

    def segment_reorder(self):
          centralSegment = self.sinogram.nSegments // 2
          o = np.zeros([self.sinogram.nSegments], dtype='int16')
          o[0::2] = np.arange(centralSegment, self.sinogram.nSegments)
          o[1::2] = np.arange(centralSegment-1, -1, -1)
          return o

    def iSSRB(self, sino2d):
          nPlanePerSeg = self.sinogram.numberOfPlanesPerSeg[self.segment_reorder()]
          mo = np.cumsum(nPlanePerSeg)
          no = np.zeros((self.sinogram.nSegments, 2), dtype='int16')
          no[:, 1] = mo
          no[1:, 0] = mo[:-1]
          sino3d = np.zeros(self.sinogram.shape, dtype='float32')
          sino3d[:, :, no[0,0]:no[0,1]] = sino2d
          for i in range(1, self.sinogram.nSegments, 2):
              delta = (nPlanePerSeg[0] - nPlanePerSeg[i]) // 2
              indx = nPlanePerSeg[0] - delta
              sino3d[:, :, no[i,0]:no[i,1]]   = sino2d[:, :, delta:indx]
              sino3d[:, :, no[i+1,0]:no[i+1,1]] = sino2d[:, :, delta:indx]
          return sino3d

    def read_sino(self, flname, num_planes=None, dtype='float32'):
         if num_planes is None:
              num_planes = self.sinogram.totalNumberOfSinogramPlanes
         sino_size = [self.sinogram.nRadialBins_orig, self.sinogram.nAngularBins, num_planes]
         return np.fromfile(flname, dtype, np.prod(sino_size)).reshape(sino_size, order='F')

    def get_e7sino(self, path):
         prompts = self.crop_sino(self.read_sino(path + self.engine.bar + 'emis_00.s'))
         randoms = self.crop_sino(self.read_sino(path + self.engine.bar + 'smoothed_rand_00.s'))
         ncf = self.crop_sino(self.read_sino(path + self.engine.bar + 'norm3d_00.a')) * self.get_gaps()
         acf = self.crop_sino(self.read_sino(path + self.engine.bar + 'acf_00.a'))
         scatters = self.iSSRB(self.crop_sino(self.read_sino(
             path + self.engine.bar + 'scatter_estim2d_000000.s', 2*self.scanner.nCrystalRings-1)))
         NF = 1 / ncf
         NF[np.isinf(NF)] = 0
         AN = (1 / acf) * NF
         RS = ncf * acf * (randoms + NF * scatters)
         return prompts, AN, RS
