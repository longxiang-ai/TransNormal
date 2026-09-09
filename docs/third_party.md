# Third-party components

The training workflow and geometry data preprocessing build on [Lotus](https://github.com/EnVision-Research/Lotus), distributed under Apache-2.0. The preprocessing code has been reorganized and reduced to the final normal-estimation workflow. A copy of the Apache-2.0 license is included in [`LICENSE-APACHE-2.0.txt`](../LICENSE-APACHE-2.0.txt); applicable upstream terms remain in effect for adapted code.

Hypersim normal orientation follows the camera-ray alignment used in Lotus and [GeoWizard](https://github.com/fuxiao0719/GeoWizard). Virtual KITTI normals use local plane fitting as in the Lotus preprocessing utilities.

DINOv3 weights, Lotus weights, and each training dataset retain their providers' licenses and access conditions. They are downloaded separately and are not bundled with the source repository. The repository's existing license remains unchanged.
