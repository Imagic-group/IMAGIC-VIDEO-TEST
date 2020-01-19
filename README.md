# IMAGIC-VIDEO-TEST

### Requirements

* CMake
* gPhoto2
* OpenCV

### Build

```bash
> mkdir build
> cd build
> cmake ..
> make
```

### Run

```bash
> gphoto2 --capture-movie --stdout | ./video ../TestData/bg.jpg
```