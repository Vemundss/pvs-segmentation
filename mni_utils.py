import os
import numpy as np
import pandas as pd
import ants
import nibabel as nib
from nipype.interfaces.ants import Registration
import tqdm


def export_data_to_nnUNet_dir(study_dir, nnUNet_dir):
    """
    Export the data in a study directory to the nnUNet directory structure.

    Parameters
    ----------
    study_dir : str
        Path to the study directory.
    nnUNet_dir : str
        Path to the nnUNet directory.

    Returns
    -------
    None

    Examples
    --------
    >>> export_data_to_nnUNet_dir("path/to/study_dir", "path/to/nnUNet_base_dir")
    """
    image_types = {"T1": "0000", "T2": "0001", "HyperT2": "0002", "FLAIR": "0003"}
    patient_dirs = [
        d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))
    ]
    patient_dirs = sorted(patient_dirs)
    i = 1
    for patient_dir in tqdm.tqdm(patient_dirs):
        visit_dirs = [
            d
            for d in os.listdir(os.path.join(study_dir, patient_dir))
            if os.path.isdir(os.path.join(study_dir, patient_dir, d))
        ]
        visit_dirs = sorted(visit_dirs)
        # print(f"Found {len(visit_dirs)} visits for patient {patient_dir}.")
        for visit_dir in visit_dirs:
            data_dir = os.path.join(study_dir, patient_dir, visit_dir)
            i_str = str(i).zfill(3)
            patient_id = patient_dir.split("_")[0]
            visit_id = visit_dir.split("_")[0]
            # copy the mask to the nnUNet directory
            mask_path = os.path.join(data_dir, "PVS_wm_to_MNI_Warped.nii.gz")
            nnunet_label_path = os.path.join(
                nnUNet_dir, "labelsTr", f"{patient_id}_{visit_id}_{i_str}.nii.gz"
            )
            command_code = os.system(f"cp {mask_path} {nnunet_label_path}")
            if command_code != 0:
                print(f"Error copying mask (does not exits?): {mask_path}")
                # skip the patient if the mask does not exist
                continue
            i += 1
            for image_type in image_types:
                # copy the image to the nnUNet directory
                image_path = os.path.join(
                    data_dir, f"{image_type}_to_MNI_Warped.nii.gz"
                )
                nnunet_image_path = os.path.join(
                    nnUNet_dir,
                    "imagesTr",
                    f"{patient_id}_{visit_id}_{i_str}_{image_types[image_type]}.nii.gz",
                )
                os.system(f"cp {image_path} {nnunet_image_path}")
            print(f"Patient {patient_id}, visit {visit_id} exported.")


def register_study_to_mni(study_dir, mask_name="PVS_wm.nii.gz", overwrite=False):
    """
    Register all images in a study to MNI space. The study directory should have the following structure:

    study_dir
    ├── patient1
    │   ├── visit1
    │   │   ├── T1.nii
    │   │   ├── T2.nii
    │   │   ├── HyperT2.nii
    │   │   └── FLAIR.nii
    │   ├── visit2
    │   │   ├── T1.nii
    │   │   ├── ...
    │   └── ...
    ├── patient2
    │   ├── ...
    └── ...

    Parameters
    ----------
    study_dir : str
        Path to the study directory.
    mask_name : str
        Name of the mask file to warp to MNI space.

    Returns
    -------
    None

    Examples
    --------
    >>> register_study_to_mni("path/to/study_dir")
    """
    image_types = ["T1", "T2", "HyperT2", "FLAIR"]
    patient_dirs = [
        d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))
    ]
    patient_dirs = sorted(patient_dirs)
    #print(f"Found {len(patient_dirs)} patients.")
    for patient_dir in tqdm.tqdm(patient_dirs):
        visit_dirs = [
            d
            for d in os.listdir(os.path.join(study_dir, patient_dir))
            if os.path.isdir(os.path.join(study_dir, patient_dir, d))
        ]
        visit_dirs = sorted(visit_dirs)
        #print(f"Found {len(visit_dirs)} visits for patient {patient_dir}.")
        for visit_dir in visit_dirs:
            data_dir = os.path.join(study_dir, patient_dir, visit_dir)
            # load the mask
            mask_path = os.path.join(data_dir, mask_name)
            if os.path.exists(mask_path):
                mask = ants.image_read(mask_path)
            else:
                print(f"Mask not found: {mask_path}")
                mask = None
            for image_type in image_types:
                mni_utils = MNIUtils(data_dir, image_type)
                mni_utils.register_to_mni(overwrite=overwrite)
                if os.path.exists(mask_path) and ants_equal_metadata(mni_utils.data, mask, verbose=False):
                    mni_utils.apply_transforms_image(
                        mask,
                        out_path=mask_path.replace(".nii.gz", "_to_MNI_Warped.nii.gz"),
                        interpolator="nearestNeighbor",
                        whichtoinvert=[False, False], # this is so stuuuupid... why is this not the default?
                    )
    return None


def ants_equal_metadata(image1, image2, verbose=True):
    """
    Check if two ANTs images have the same metadata.

    Parameters
    ----------
    image1 : ants.core.ants_image.ANTsImage
        First image.
    image2 : ants.core.ants_image.ANTsImage
        Second image.
    verbose : bool
        Print verbose output.

    Returns
    -------
    bool
        True if the metadata is equal, False otherwise.

    Examples
    --------
    >>> ants_equal_metadata(image1, image2)
    """
    if not isinstance(image1, ants.core.ants_image.ANTsImage):
        raise ValueError(r"image1 must be an ANTs image. But got: {type(image1)}")
    if not isinstance(image2, ants.core.ants_image.ANTsImage):
        raise ValueError(r"image2 must be an ANTs image. But got: {type(image2)}")
    if image1.shape != image2.shape:
        if verbose:
            print(f"Shape mismatch: {image1.shape} != {image2.shape}")
        return False
    if image1.origin != image2.origin:
        if verbose:
            print(f"Origin mismatch: {image1.origin} != {image2.origin}")
        return False
    if image1.spacing != image2.spacing:
        if verbose:
            print(f"Spacing mismatch: {image1.spacing} != {image2.spacing}")
        return False
    if (image1.direction != image2.direction).all():
        if verbose:
            print(f"Direction mismatch: {image1.direction} != {image2.direction}")
        return False
    return True


class MNIUtils:
    def __init__(self, data_dir, image_name):
        self.data_dir = data_dir
        self.image_name = image_name
        self.image_path = os.path.join(self.data_dir, f"{self.image_name}.nii")
        self.load_as = "ants"  # or 'nibabel'

    @property
    def data(self):
        """Load the native space image."""
        return (
            ants.image_read(os.path.join(self.data_dir, f"{self.image_name}.nii"))
            if self.load_as == "ants"
            else nib.load(self.image_path)
        )

    @property
    def mnidata(self):
        """Load the MNI warped image if it exists."""
        return (
            ants.image_read(
                os.path.join(self.data_dir, f"{self.image_name}_to_MNI_Warped.nii.gz")
            )
            if self.load_as == "ants"
            else nib.load(
                os.path.join(self.data_dir, f"{self.image_name}_to_MNI_Warped.nii.gz")
            )
        )

    @property
    def affine_path(self):
        return os.path.join(
            self.data_dir, f"{self.image_name}_to_MNI_0GenericAffine.mat"
        )

    @property
    def warp_path(self):
        return os.path.join(self.data_dir, f"{self.image_name}_to_MNI_1Warp.nii.gz")

    @property
    def inverse_warp_path(self):
        return os.path.join(
            self.data_dir, f"{self.image_name}_to_MNI_1InverseWarp.nii.gz"
        )

    @property
    def inverse_affine(self):
        affine = ants.read_transform(self.affine_path)
        return ants.invert_transform(affine)

    def register_to_mni(self, overwrite=False, **kwargs):
        """Register the image to MNI space."""
        if self._is_registered_to_mni() and not overwrite:
            print("Image is already registered to MNI space.")
            return
        fixed = ants.image_read(ants.get_ants_data("mni"))
        try:
            moving = ants.image_read(
                os.path.join(self.data_dir, f"{self.image_name}.nii")
            )
        except:
            print(f"Could not load image: {self.image_path}")
            return
        outprefix = os.path.join(self.data_dir, f"{self.image_name}_to_MNI_")
        # Run registration
        registration_result = ants.registration(
            fixed=fixed, moving=moving, outprefix=outprefix, **kwargs
        )
        # Save the warped moving image
        #warped_image_path = f"{outprefix}Warped.nii.gz"
        #ants.image_write(registration_result["warpedmovout"], warped_image_path)
        #print(f"Warped moving image saved to: {warped_image_path}")
        return registration_result

    def _is_registered_to_mni(self):
        return os.path.exists(
            os.path.join(self.data_dir, f"{self.image_name}_to_MNI_Warped.nii.gz")
        )

    def apply_transforms_coords(self, coords, transform_paths):
        """Apply transforms to coordinates using apply_transforms_to_points."""
        if isinstance(coords, tuple):
            coords = np.array([coords])
        points_df = pd.DataFrame(coords, columns=["x", "y", "z"])
        transformed_points_df = ants.apply_transforms_to_points(
            3, points_df, transform_paths
        )
        return transformed_points_df[["x", "y", "z"]].values

    def apply_transforms_image(self, image, out_path=None, **kwargs):
        """
        Apply the affine and warp transforms to an image (or mask).

        Parameters
        ----------
        image_path : str or ants.core.ants_image.ANTsImage
            Path to the image to transform.
        out_path : str

        Returns
        -------
        transformed_image : ants.core.ants_image.ANTsImage
            The transformed image.
        """
        if isinstance(image, str):
            image_path = image
            image = ants.image_read(image_path)
        whichtoinvert = [False, False] if "whichtoinvert" not in kwargs else kwargs["whichtoinvert"]
        kwargs.pop("whichtoinvert", None)
        transformed_image = ants.apply_transforms(
            fixed=ants.image_read(ants.get_ants_data("mni")),
            moving=image,
            transformlist=[self.affine_path, self.warp_path],
            whichtoinvert=whichtoinvert,
            **kwargs,
        )
        if out_path is not None:
            ants.image_write(transformed_image, out_path)
            print(f"Transformed image saved to: {out_path}")
        return transformed_image
