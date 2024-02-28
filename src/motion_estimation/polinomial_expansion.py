'''Farneback's polinomial expansion. See https://github.com/ericPrince/optical-flow'''

import numpy as np
import logging
#logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

class Expander():

    def __init__(self, verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)

    def expand_1D(self, f, c, sigma):
        """
        Calculates the local polynomial expansion of a 1D signal.
        
        $f ~ x^T A x + B^T x + C$
        
        If f[i] and c[i] are the signal value and certainty of sample i then
        A[i] is a 1x1 array representing the quadratic term of the polynomial, B[i]
        is a 1-element array representing the linear term, and C[i] is a scalar
        representing the constant term.
        
        Parameters
        ----------
        f
            Input signal
        c
            Certainty of signal
        sigma
            Standard deviation of applicability Gaussian kernel
        Returns
        -------
        A
            Quadratic term of polynomial expansion
        B
            Linear term of polynomial expansion
        C
            Constant term of polynomial expansion
        """

        # Kernel applicability (gaussian)
        n = int(4 * sigma + 1)
        x = np.arange(-n, n + 1, dtype=np.int32)
        a = np.exp(-(x**2) / (2 * sigma**2))
    
        # b: calculate b from the paper.
        b = np.stack([np.ones(a.shape), x, x**2], axis=-1)
    
        # Pre-calculate product of certainty and signal
        cf = c * f
    
        # G and v are used to calculate "r" from the paper: v = G*r
        # r is the parametrization of the 2nd order polynomial for f
        G = np.empty(list(f.shape) + [b.shape[-1]] * 2)
        v = np.empty(list(f.shape) + [b.shape[-1]])

        # Apply cross-correlation
    
        # Pre-calculate quantities recommended in paper
        ab = np.einsum("i,ij->ij", a, b) # ab[i] = b[i]*a[i]
        abb = np.einsum("ij,ik->ijk", ab, b) # abb[i,j] = ab[i]*b[j]
    
        # Calculate G and v for each pixel with cross-correlation
        for i in range(b.shape[-1]):
            for j in range(b.shape[-1]):
                #print("G[..., i, j].shape", G[..., i, j].shape)
                G[..., i, j] = scipy.ndimage.correlate1d(
                    c, abb[..., i, j], axis=0, mode="constant", cval=0
                )
    
            v[..., i] = scipy.ndimage.correlate1d(
                cf, ab[..., i], axis=0, mode="constant", cval=0
            )
    
        # Solve r for each pixel
        r = np.linalg.solve(G, v)
    
        # Quadratic term
        A = np.empty(list(f.shape) + [1, 1])
        A[..., 0, 0] = r[..., 2]
    
        # Linear term
        B = np.empty(list(f.shape) + [1])
        B[..., 0] = r[..., 1]
    
        # constant term
        C = r[..., 0]
    
        return A, B, C

    def expand_2D(self, f, c, sigma):
        """
        Calculates the local polynomial expansion of a 2D signal, as described by Farneback
        Uses separable normalized correlation
        $f ~ x^T A x + B^T x + C$
        If f[i, j] and c[i, j] are the signal value and certainty of pixel (i, j) then
        A[i, j] is a 2x2 array representing the quadratic term of the polynomial, B[i, j]
        is a 2-element array representing the linear term, and C[i, j] is a scalar
        representing the constant term.
        Parameters
        ----------
        f
            Input signal
        c
            Certainty of signal
        sigma
            Standard deviation of applicability Gaussian kernel
        Returns
        -------
        A
            Quadratic term of polynomial expansion
        B
            Linear term of polynomial expansion
        C
            Constant term of polynomial expansion
        """
        # Calculate applicability kernel (1D because it is separable)
        n = int(4 * sigma + 1)
        x = np.arange(-n, n + 1, dtype=np.int32)
        a = np.exp(-(x**2) / (2 * sigma**2))  # a: applicability kernel [n]
    
        # b: calculate b from the paper. Calculate separately for X and Y dimensions
        # [n, 6]
        bx = np.stack(
            [np.ones(a.shape), x, np.ones(a.shape), x**2, np.ones(a.shape), x], axis=-1
        )
        by = np.stack(
            [np.ones(a.shape), np.ones(a.shape), x, np.ones(a.shape), x**2, x, ], axis=-1,
        )
    
        # Pre-calculate product of certainty and signal
        cf = c * f
    
        # G and v are used to calculate "r" from the paper: v = G*r (see Eq. 4.9 of the thesis)
        # r is the parametrization of the 2nd order polynomial for f
        G = np.empty(list(f.shape) + [bx.shape[-1]] * 2)
        v = np.empty(list(f.shape) + [bx.shape[-1]])
    
        # Apply separable cross-correlations
    
        # Pre-calculate quantities recommended in paper (Eq. 4.6 in the thesis)
        ab = np.einsum("i,ij->ij", a, bx) # ab[i,j] = bx[i,j]*a[i] (multiply each row of "bx" by the corresponding element of "a")
        abb = np.einsum("ij,ik->ijk", ab, bx) # abb[i,j,k] = ab[i,j]*bx[j,k]
        
        # Calculate G and v for each pixel with cross-correlation (axis 0)
        #print("bx.shape[-1]", bx.shape[-1])
        for i in range(bx.shape[-1]):
            for j in range(bx.shape[-1]):
                #print("G[..., i, j].shape", G[..., i, j].shape)
                G[..., i, j] = scipy.ndimage.correlate1d(
                    c, abb[..., i, j], axis=0, mode="constant", cval=0
                )
    
            v[..., i] = scipy.ndimage.correlate1d(
                cf, ab[..., i], axis=0, mode="constant", cval=0
            )
    
        # Pre-calculate quantities recommended in paper
        ab = np.einsum("i,ij->ij", a, by)
        abb = np.einsum("ij,ik->ijk", ab, by)
    
        # Calculate G and v for each pixel with cross-correlation (axis 1)
        for i in range(bx.shape[-1]):
            for j in range(bx.shape[-1]):
                G[..., i, j] = scipy.ndimage.correlate1d(
                    G[..., i, j], abb[..., i, j], axis=1, mode="constant", cval=0
                )
    
            v[..., i] = scipy.ndimage.correlate1d(
                v[..., i], ab[..., i], axis=1, mode="constant", cval=0
            )
    
        # Solve r for each pixel (eq. 4.8 of the thesis)
        r = np.linalg.solve(G, v)
    
        # Quadratic term
        A = np.empty(list(f.shape) + [2, 2])
        A[..., 0, 0] = r[..., 3]
        A[..., 0, 1] = r[..., 5] / 2
        A[..., 1, 0] = A[..., 0, 1]
        A[..., 1, 1] = r[..., 4]
    
        # Linear term
        B = np.empty(list(f.shape) + [2])
        B[..., 0] = r[..., 1]
        B[..., 1] = r[..., 2]
    
        # constant term
        C = r[..., 0]
    
        return A, B, C
        
class Filter_Monochrome_Image(motion_estimation.farneback.Estimator_in_CPU):
    def __init__(self, l=3, w=15, OF_iters=3, poly_n=5, poly_sigma=1.0, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)
        super().__init__(levels=l, win_side=w, iters=OF_iters, poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)
        self.logger.info(f"l={l}")
        self.logger.info(f"w={w}")
        self.logger.info(f"OF_iters={OF_iters}")
        self.logger.info(f"poly_n={poly_n}")
        self.logger.info(f"poly_sigma={poly_sigma}")
        self.logger.info(f"flags={flags}")

    def project_A_to_B(self, A, B):
        #flow = self.get_flow_to_project_A_to_B(A, B)
        flow = self.get_flow(target=B, reference=A, prev_flow=None)
        flow = np.zeros_like(flow)
        self.logger.info(f"np.average(np.abs(flow))={np.average(np.abs(flow))}")
        #return flow_estimation.project(A, flow)
        projection = motion_estimation.helpers.project(image=A, flow=flow)
        return projection

    def normalize(self, img):
        min_img = np.min(img)
        max_img = np.max(img)
        return 255*((img - min_img + 1)/(max_img - min_img + 1))

    def randomize(self, image, mean=0, std_dev=1.0):
        height, width = image.shape[:2]
        self.logger.debug(f"image.shape={image.shape}")
        x_coords, y_coords = np.meshgrid(range(width), range(height)) # Create a grid of coordinates
        flattened_x_coords = x_coords.flatten()
        flattened_y_coords = y_coords.flatten()
        displacements_x = np.random.normal(mean, std_dev, flattened_x_coords.shape)
        displacements_y = np.random.normal(mean, std_dev, flattened_y_coords.shape)
        #displacements_x *= max_distance_x # Scale the displacements by the maximum distance
        #displacements_y *= max_distance_y
        displacements_x = displacements_x.astype(np.int32)
        displacements_y = displacements_y.astype(np.int32)

        self.logger.info(f"np.average(np.abs(displacements_x))={np.average(np.abs(displacements_x))} np.average(np.abs(displacements_y))={np.average(np.abs(displacements_y))}")
        self.logger.info(f"np.max(displacements_x)={np.max(displacements_x)} np.max(displacements_y)={np.max(displacements_y)}")
        self.logger.info(f"np.min(displacements_x)={np.min(displacements_x)} np.min(displacements_y)={np.min(displacements_y)}")
        randomized_x_coords = flattened_x_coords + displacements_x
        randomized_y_coords = flattened_y_coords + displacements_y
        randomized_x_coords = np.clip(randomized_x_coords, 0, width - 1) # Clip the randomized coordinates to stay within image bounds
        randomized_y_coords = np.clip(randomized_y_coords, 0, height - 1)
        #randomized_x_coords = np.mod(randomized_x_coords, width) # Apply periodic extension to handle border pixels
        #randomized_y_coords = np.mod(randomized_y_coords, height)
        #randomized_image = np.ones_like(image) * np.average(image)
        randomized_image = np.zeros_like(image)
        #randomized_image[...] = image
        #randomized_image[...] = 255
        randomized_image[randomized_y_coords, randomized_x_coords] = image[flattened_y_coords, flattened_x_coords]
        return randomized_image

    def _randomize(self, image, max_distance=10):
        height, width = image.shape[:2]
        #flow_x = np.random.normal(loc=0, scale=std_dev, size=(height, width))
        flow_x = np.random.normal(size=(height, width)) * max_distance
        flow_y = np.random.normal(size=(height, width)) * max_distance
        #flow_x = np.random.uniform(low=-1, high=1, size=(height, width)) * max_distance
        #flow_y = np.random.uniform(low=-1, high=1, size=(height, width)) * max_distance
        #flow_x[...] = 0
        #flow_y[...] = 0
        #print(np.max(flow_x), np.min(flow_x), max_distance)
        flow = np.empty([height, width, 2], dtype=np.float32)
        flow[..., 0] = flow_y
        flow[..., 1] = flow_x
        print(np.max(flow), np.min(flow))
        randomized_image = motion_estimation.project(image, flow)
        return randomized_image.astype(np.uint8)

    def _randomize(self, image, max_distance=150):
        noise = np.random.normal(0, max_distance, image.shape).reshape(image.shape)
        return np.clip(a=image.astype(np.float32) + noise, a_min=0, a_max=255).astype(np.uint8)

    def compute_quality_index(self, img, denoised_img):
        diff_img = (img - denoised_img).astype(np.uint8)
        _, N = ssim(img, diff_img, full=True)
        _, P = ssim(img, denoised_img.astype(np.uint8), full=True)
        quality, _ = stats.pearsonr(N.flatten(), P.flatten())
        if math.isnan(quality):
            return 0.0
        else:
            return -quality

    def filter(self,
               noisy_image,
               GT=None,
               N_iters=50,
               RS_sigma=1.0, # Standard deviation of the maximum random (gaussian-distributed) displacements of the pixels
               RS_mean=0.0, # Mean of the randomized distances
               #RD_sigma=1.0,
               #levels=3,
               #window_side=2,
               #poly_n=5,
               #poly_sigma=0.3,
               ):

        #logger.info(f"RD_iters={RD_iters} RD_mean={RD_mean} RD_sigma={sigma} levels={levles} window_side={window_side} poly_n={poly_n} poly_sigma={poly_sigma}")
        self.logger.info(f"N_iters={N_iters} RS_mean={RS_mean} RS_sigma={RS_sigma}")
        if self.logger.level <= logging.INFO:
            PSNR_vs_iteration = []

        acc_image = np.zeros_like(noisy_image, dtype=np.float32)
        acc_image[...] = noisy_image
        if self.logger.level <= logging.DEBUG:
            denoised_image = noisy_image
        current_QI = -1.0
        for i in range(N_iters):
            self.logger.info(f"Iteration {i}/{N_iters}")
            if self.logger.level <= logging.DEBUG:
                fig, axs = plt.subplots(1, 2, figsize=(16, 32))
                prev = denoised_image
            denoised_image = acc_image/(i+1)
            if self.logger.level <= logging.INFO:
                try:
                    _PSNR = information_theory.distortion.PSNR(denoised_image, GT)
                except:
                    _PSNR = 0.0
                PSNR_vs_iteration.append(_PSNR)
            if self.logger.level <= logging.DEBUG:
                axs[0].imshow(denoised_image.astype(np.uint8))
                axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
                #axs[1].imshow(self.normalize(noisy_image - denoised_image + 128).astype(np.uint8), cmap="gray")
                axs[1].imshow((noisy_image.astype(np.float32) - denoised_image), cmap="gray")
                axs[1].set_title(f"diff")
                plt.show()
            #randomized_noisy_image = self._randomize(noisy_image, max_distance=50)
            randomized_noisy_image = self.randomize(noisy_image, mean=0, std_dev=RS_sigma)
            #randomized_noisy_image = randomize(noisy_image)
            randomized_and_compensated_noisy_image = self.project_A_to_B(
                A=denoised_image,
                B=randomized_noisy_image)
            acc_image += randomized_and_compensated_noisy_image
            prev_QI = current_QI
            current_QI = self.compute_quality_index(noisy_image, denoised_image)
            self.logger.info(f"prev_QI={prev_QI} current_QI={current_QI}")
            #if current_QI < prev_QI:
            #    break
        denoised_image = acc_image/(N_iters + 1)
        #print(flush=True)

        if self.logger.level <= logging.INFO:
            return denoised_image, PSNR_vs_iteration
        else:
            return denoised_image, None

class Filter_Color_Image(Filter_Monochrome_Image):

    def __init__(
            self,
            l=3, # Pyramid slope. Multiply by 2^levels the searching area if the OFE
            w=15, # Applicability window side
            OF_iters=3, # Number of iterations at each pyramid level
            poly_n=5, # Size of the pixel neighborhood used to find polynomial expansion in each pixel
            poly_sigma=1.0, # Standard deviation of the Gaussian basis used in the polynomial expansion
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            verbosity=logging.INFO):
        super().__init__(l, w, OF_iters, poly_n, poly_sigma, flags, verbosity)

    def project_A_to_B(self, A, B):
        self.logger.debug(f"A.shape={A.shape} B.shape={B.shape}")
        A_luma = YUV.from_RGB(A.astype(np.int16))[..., 0]
        B_luma = YUV.from_RGB(B.astype(np.int16))[..., 0]
        #A_luma = np.log(YUV.from_RGB(A.astype(np.int16))[..., 0] + 1)
        #B_luma = np.log(YUV.from_RGB(B.astype(np.int16))[..., 0] + 1)
        #flow = self.get_flow_to_project_A_to_B(A_luma, B_luma)
        flow = self.get_flow(target=B_luma, reference=A_luma, prev_flow=None)
        self.logger.info(f"np.average(np.abs(flow))={np.average(np.abs(flow))}")
        #return flow_estimation.project(A, flow)
        #return super().warp_B_to_A(A_luma,
        #                           B_luma)
        projection = motion_estimation.helpers.project(image=A, flow=flow)
        return projection

    def compute_quality_index(self, img, denoised_img):
        Y_img = YUV.from_RGB(img)[..., 0]
        Y_denoised_img = YUV.from_RGB(denoised_img)[..., 0]
        diff_img = (Y_img - Y_denoised_img).astype(np.uint8)
        _, N = ssim(Y_img, diff_img, full=True)
        _, P = ssim(Y_img, Y_denoised_img.astype(np.uint8), full=True)
        quality, _ = stats.pearsonr(N.flatten(), P.flatten())
        if math.isnan(quality):
            return 0.0
        else:
            return -quality
'''
class Filter_Monochrome_Image_OLD(flow_estimation.Farneback_Flow_Estimator):

    def __init__(
            self,
            l=3,       # Pyramid slope. Multiply by 2^levels the searching area if the OFE
            w=15, # Applicability window side
            OF_iters=3,     # Number of iterations at each pyramid level
            poly_n=5,       # Size of the pixel neighborhood used to the find polynomial expansion in each pixel
            poly_sigma=1.0, # Standard deviation of the Gaussian basis used in the polynomial expansion
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)
        print(f"logging level = {self.logger.level}")
        super().__init__(l, w, OF_iters, poly_n, poly_sigma, flags)
        self.logger.debug(f"l={l}, w={w}, OF_iters={OF_iters}, poly_n={poly_n}, poly_sigma={poly_sigma}")

    def project_A_to_B(self, A, B):
        #flow = self.get_flow_to_project_A_to_B(A, B)
        flow = self.get_flow(target=B, reference=A, prev_flow=None)
        self.logger.info(f"np.average(np.abs(flow))={np.average(np.abs(flow))}")
        return flow_estimation.project(A, flow)

    def normalize(self, img):
        min_img = np.min(img)
        max_img = np.max(img)
        return 255*((img - min_img + 1)/(max_img - min_img + 1))

    def randomize(self, image, mean=0, std_dev=1.0):
        height, width = image.shape[:2]
        self.logger.debug(f"image.shape={image.shape}")
        x_coords, y_coords = np.meshgrid(range(width), range(height)) # Create a grid of coordinates
        flattened_x_coords = x_coords.flatten()
        flattened_y_coords = y_coords.flatten()
        displacements_x = np.random.normal(mean, std_dev, flattened_x_coords.shape)
        displacements_y = np.random.normal(mean, std_dev, flattened_y_coords.shape)
        #displacements_x *= max_distance_x # Scale the displacements by the maximum distance
        #displacements_y *= max_distance_y
        displacements_x = displacements_x.astype(np.int32)
        displacements_y = displacements_y.astype(np.int32)

        self.logger.info(f"np.max(displacements_x)={np.max(displacements_x)} np.max(displacements_y)={np.max(displacements_y)}")
        randomized_x_coords = flattened_x_coords + displacements_x
        randomized_y_coords = flattened_y_coords + displacements_y
        #randomized_x_coords = np.clip(randomized_x_coords, 0, width - 1) # Clip the randomized coordinates to stay within image bounds
        #randomized_y_coords = np.clip(randomized_y_coords, 0, height - 1)
        randomized_x_coords = np.mod(randomized_x_coords, width) # Apply periodic extension to handle border pixels
        randomized_y_coords = np.mod(randomized_y_coords, height)
        randomized_image = np.zeros_like(image)
        randomized_image[randomized_y_coords, randomized_x_coords] = image[flattened_y_coords, flattened_x_coords]
        return randomized_image

    def filter(self,
               noisy_image,
               GT=None,
               N_iters=50,
               RS_sigma=1.0, # Standard deviation of the maximum random (gaussian-distributed) displacements of the pixels
               RS_mean=0.0, # Mean of the randomized distances
               #RD_sigma=1.0,
               #levels=3,
               #window_side=2,
               #poly_n=5,
               #poly_sigma=0.3,
               ):

        #logger.info(f"RD_iters={RD_iters} RD_mean={RD_mean} RD_sigma={sigma} levels={levles} window_side={window_side} poly_n={poly_n} poly_sigma={poly_sigma}")
        self.logger.info(f"N_iters={N_iters} RS_mean={RS_mean} RS_sigma={RS_sigma}")
        if self.logger.level <= logging.INFO:
            PSNR_vs_iteration = []

        acc_image = np.zeros_like(noisy_image, dtype=np.float32)
        acc_image[...] = noisy_image
        if self.logger.level <= logging.DEBUG:
            denoised_image = noisy_image
        for i in range(N_iters):
            self.logger.info(f"Iteration {i}/{N_iters}")
            if self.logger.level <= logging.DEBUG:
                fig, axs = plt.subplots(1, 2)
                prev = denoised_image
            denoised_image = acc_image/(i+1)
            if self.logger.level <= logging.INFO:
                try:
                    _PSNR = information_theory.distortion.PSNR(denoised_image, GT)
                except:
                    _PSNR = 0.0
                PSNR_vs_iteration.append(_PSNR)
            if self.logger.level <= logging.DEBUG:
                axs[0].imshow(denoised_image.astype(np.uint8))
                axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
                axs[1].imshow(self.normalize(prev - denoised_image + 128).astype(np.uint8), cmap="gray")
                axs[1].set_title(f"diff")
                plt.show()
            randomized_noisy_image = self.randomize(
                noisy_image,
                RS_mean,
                RS_sigma).astype(np.float32)
            #randomized_noisy_image = randomize(noisy_image)
            randomized_and_compensated_noisy_image = self.project_A_to_B(
                A=denoised_image,
                B=randomized_noisy_image)
            acc_image += randomized_and_compensated_noisy_image
        denoised_image = acc_image/(N_iters + 1)
        #print(flush=True)

        if self.logger.level <= logging.INFO:
            return denoised_image, PSNR_vs_iteration
        else:
            return denoised_image, None
'''


'''
def _randomize(image, max_distance_x=10, max_distance_y=10):
    height, width, _ = image.shape
    x_coords, y_coords = np.meshgrid(range(width), range(height)) # Create a grid of coordinates
    flattened_x_coords = x_coords.flatten()
    flattened_y_coords = y_coords.flatten()
    displacements_x = np.random.randint(-max_distance_x, max_distance_x + 1, flattened_x_coords.shape)
    displacements_y = np.random.randint(-max_distance_y, max_distance_y + 1, flattened_y_coords.shape)
    randomized_x_coords = flattened_x_coords + displacements_x
    randomized_y_coords = flattened_y_coords + displacements_y
    randomized_x_coords = np.mod(randomized_x_coords, width) # Use periodic extension to handle border pixels
    randomized_y_coords = np.mod(randomized_y_coords, height)
    randomized_image = np.empty_like(image)
    randomized_image[...] = image
    randomized_image[randomized_y_coords,
                     randomized_x_coords, :] = image[flattened_y_coords,
                                                     flattened_x_coords, :]
    return randomized_image

def RGB_warp_B_to_A(A, B, l=3, w=15, prev_flow=None, sigma=1.5):
    A_luma = YUV.from_RGB(A.astype(np.int16))[..., 0]
    B_luma = YUV.from_RGB(B.astype(np.int16))[..., 0]
    #A_luma = np.log(YUV.from_RGB(A.astype(np.int16))[..., 0] + 1)
    #B_luma = np.log(YUV.from_RGB(B.astype(np.int16))[..., 0] + 1)
    flow = flow_estimation.get_flow_to_project_A_to_B(A_luma, B_luma, l, w, prev_flow, sigma)
    return flow_estimation.project(B, flow)

def warp_B_to_A(A, B, l=3, w=15, prev_flow=None, sigma=1.5):
    flow = flow_estimation.get_flow_to_project_A_to_B(A, B, l, w, prev_flow, sigma)
    return flow_estimation.project(B, flow)
                                   
def filter_image(
        warp_B_to_A,
        noisy_image,
        N_iters=50,
        mean_RD=0.0,
        sigma_RD=1.0,
        l=3,
        w=2,
        sigma_OF=0.3,
        GT=None):

    logger.info(f"N_iters={N_iters} mean_RD={mean_RD} sigma_RD={sigma_RD} l={l} w={w} sigma_OF={sigma_OF}")
    if logger.level <= logging.INFO:
        PSNR_vs_iteration = []

    acc_image = np.zeros_like(noisy_image, dtype=np.float32)
    acc_image[...] = noisy_image
    denoised_image = noisy_image
    for i in range(N_iters):
        print(f"{i}/{N_iters}", end=' ')
        if logger.level <= logging.DEBUG:
            fig, axs = plt.subplots(1, 2)
            prev = denoised_image
        denoised_image = acc_image/(i+1)
        if logger.level <= logging.DEBUG:
            if GT != None:
                _PSNR = information_theory.distortion.PSNR(denoised_image, GT)
            else:
                _PSNR = 0.0
            PSNR_vs_iteration.append(_PSNR)
            axs[0].imshow(denoised_image.astype(np.uint8))
            axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
            axs[1].imshow(normalize(prev - denoised_image + 128).astype(np.uint8), cmap="gray")
            axs[1].set_title(f"diff")
            plt.show()
        randomized_noisy_image = randomize(noisy_image, mean_RD, sigma_RD).astype(np.float32)
        randomized_and_compensated_noisy_image = warp_B_to_A(
            A=randomized_noisy_image,
            B=denoised_image,
            l=l,
            w=w,
            sigma=sigma_OF)
        acc_image += randomized_and_compensated_noisy_image
    denoised_image = acc_image/(N_iters + 1)
    print()

    if logger.level <= logging.INFO:
        return denoised_image, PSNR_vs_iteration
    else:
        return denoised_image, None

def _denoise(warp_B_to_A, noisy_image, N_iters=50, mean_RD=0.0, sigma_RD=1.0, l=3, w=2, sigma_OF=0.3):
    logger.info(f"N_iters={N_iters} mean_RD={mean_RD} sigma_RD={sigma_RD} l={l} w={w} sigma_OF={sigma_OF}")
    acc_image = np.zeros_like(noisy_image, dtype=np.float32)
    acc_image[...] = noisy_image
    for i in range(N_iters):
        print(f"iter={i}", end=' ')
        denoised_image = acc_image/(i+1)
        randomized_noisy_image = randomize(noisy_image, mean_RD, sigma_RD).astype(np.float32)
        randomized_and_compensated_noisy_image = warp_B_to_A(A=randomized_noisy_image, B=denoised_image, l=l, w=w, sigma=sigma_OF)
        acc_image += randomized_and_compensated_noisy_image
    denoised_image = acc_image/(N_iters + 1)
    print()
    return denoised_image
'''
