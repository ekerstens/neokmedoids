from setuptools import setup

# specify requirements of your package here
REQUIREMENTS = ['numpy']

# some more details
CLASSIFIERS = [
	'Intended Audience :: Developers',
	'Topic :: Clustering',
	'License :: GNU General Public License v3.0',
	'Programming Language :: Python :: 3.7',
	]

# calling the setup function
setup(name='neokmedoids',
	version='1.0.0',
	description='Implementation of neo-k-medoids clustering algorithm',
	author='Eric Kerstens',
	author_email='kersten_eric@bentley.edu',
	license='GNU General Public License v3.0',
	classifiers=CLASSIFIERS,
	install_requires=REQUIREMENTS,
	keywords='clustering'
	)
