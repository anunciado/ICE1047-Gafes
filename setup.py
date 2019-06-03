from setuptools import setup

folder = os.path.dirname(os.path.realpath(__file__))
requirements = folder + '/requirements.txt'
install_requires = []

if os.path.isfile(requirements):
    with open(requirements) as f:
        install_requires = f.read().splitlines()

setup(
    name="gafes",
    version="0.0.1",
    author_email="anunciado@protonmail.com",
    description="A tool for feature selection using genetic algorithms. ",
    url="https://github.com/anunciado/ICE1047-Gafes",
    install_requires=install_requires,
    py_modules = ['<filename>']
)
