<img src="https://www.dropbox.com/scl/fi/viulr3lyxvlum7vhqvagk/Salford-3.png?rlkey=8jpl2coezs1kr0ue3n33dmg32&st=zjc3b3bn&raw=1" style="width: 50%;" />


# Project BTE - beyond the eyes. Mapping 3D depth maps to 3D binaural audio.
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Static Badge](https://img.shields.io/badge/Python-blue)
![MacOS](https://img.shields.io/badge/MacOS-green)

Built for my 12th grade quantitative research at Leyte National High School — STEM. 

## The Idea
________________________

The idea behind this project heavily relies on the paper by Ranftl et. al., Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer, TPAMI 2022. I utilized this [repository](https://github.com/isl-org/MiDaS) for generating depth maps based on a monocular input. I tried working on extending its capabilities towards binocular inputs but due to the time constraint, as it was also for a research competition, I wasn't able to further develop this. 

On the audio side, I am quite an audio guy (I make music!). I used _dearVR Micro_ to generate "3D" transformations of a sine wave. Since _Micro_ has 2 parameters: **elevation** and **azimuth**, I simply mapped out the x and y coordinates of the detected object and used its depth as volume. This approach is oversimplified, and could absolutely be improved much more with trigonometric derivations and such.

_On a side note, this project is quite old and I didn't mind code readability (obviously) at the time lol._

https://github.com/user-attachments/assets/c0938d8e-e30f-4e41-9623-7798eab981b2

## Usage
________________________

I will update this README with a better documentation in the future, but for now you can simply run
```bash
python midasDepthMap2.py
```

Make sure to install necessary dependencies as listed in `midasDepthMap2.py`. I will create a requirements.txt file later for better installation flow.

There are also static paths on certain files which point to absolute paths on my local environment, you might have to change it on your end.

## Areas for Improvement
________________________

- Applying the logic behind 3D perspective projection to recreate better the 3D environment
- Utilizing a better tool for 3D binaural audio (I believe dearVR's Micro might not be enough)
- Better documentation

## License
________________________

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
 