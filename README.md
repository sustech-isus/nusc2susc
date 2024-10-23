# NUSC2SUSC Converter

This tool converts the [nuScenes](https://www.nuscenes.org/) dataset to the [SUSCape](https://suscape.net) dataset format.

## Overview

The NUSC2SUSC Converter is designed to transform nuScenes data into a format compatible with SUSCape, facilitating easier integration and use with SUSCape-based projects.

## Features

Currently supported data conversions:
- LiDAR point clouds
- Camera images
- LiDAR poses
- Object tracking labels

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/nusc2susc-converter.git
   cd nusc2susc-converter
   ```

2. Install the required dependencies (a virtual env is suggested):
   ```
   pip install -r requirements.txt   
   ```

## Usage

Run the conversion script:

```sh
python convert.py --nusc_path /path/to/nuscenes/data --output_path /path/to/output
```

Use `python convert.py --help` for more options.


Check correctness with:

```sh
python visualize.py --scene-root /path/to/converted/data
```

## Roadmap

Future development plans include support for:
- Radar data (not sure if possible ...)
- HD map data
- Ego vehicle poses
- Camera calibration data
- Additional metadata and annotations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [nuScenes](https://www.nuscenes.org/) for providing the original dataset
- [SUSCape](https://suscape.net) for the target dataset format

## Contact

For questions or support, please open an issue in this repository or contact [your-email@example.com].