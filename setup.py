
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='i3d_jax',  
     version='0.0.1',
     author="Wilson Yan",
     author_email="wilson1.yan@berkeley.edu",
     description="I3D in Jax",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/wilson1yan/i3d-jax",
     packages=['i3d_jax'],
     package_data={'i3d_jax': ['weights/*.ckpt']},
     include_package_data=True,
     install_requires=['jax', 'flax'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ]
 )
