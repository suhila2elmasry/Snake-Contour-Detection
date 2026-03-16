@echo off
echo Setting up Snake Contour Project...
echo.

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Installing Django and REST framework...
pip install django==4.2.7
pip install djangorestframework==3.14.0
pip install django-cors-headers==4.3.1

echo.
echo Installing scientific packages...
pip install numpy==1.24.3
pip install scipy==1.11.4
pip install opencv-python==4.8.1.78
pip install Pillow==10.1.0
pip install matplotlib==3.8.2
pip install scikit-image==0.22.0

echo.
echo Verifying installations...
python -c "import django; print('Django:', django.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import scipy; print('SciPy:', scipy.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import PIL; print('Pillow:', PIL.__version__)"
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
python -c "import skimage; print('Scikit-image:', skimage.__version__)"

echo.
echo All packages installed successfully!
echo.
echo Now you can run the server with:
echo cd backend
echo python manage.py runserver
pause