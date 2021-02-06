from setuptools import setup

setup(name='mrsgym',
      version='1.0.0',
      install_requires=['gym','scipy','torch','pybullet','numpy'],
      include_package_data=True,
      author = 'Ryan Kortvelesy',
      author_email = 'rk627@cam.ac.uk',
      description = 'A pybullet-based multiagent simulator',
)