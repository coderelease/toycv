from setuptools import setup, find_packages


def read(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read()


setup(
    name='toycv',  # 包名
    python_requires='>=3.8.0',  # python环境
    version='0.3.3',  # 包的版本
    description="A simple and flexible tool for building and training neural network models.",  # 包简介，显示在PyPI上

    long_description=read('README.md'),  # 读取的Readme文档内容，一整块字符串
    long_description_content_type="text/markdown",  # 指定包文档格式为markdown

    # 作者相关信息
    author="Yang Xiang",
    author_email='btk@qq.com',
    url='https://github.com/coderelease',

    # 指定包信息，还可以用find_packages()函数  # find_packages(where="./", include=["toycv*"]),
    # packages=["toycv",
    #           "toycv.image_video_utils"],
    packages=find_packages(where=".", include=["toycv*"]),
    install_requires=read('requirements.txt').splitlines(),  # 指定需要安装的依赖, 需要是一个列表

    # 如果你在 setup.py 中设置了 include_package_data=True，
    # 并且在 MANIFEST.in 文件中指定了 include data/*，
    # 那么在安装包时，config.ini 和 template.html 将会被自动包含在安装的包中。
    include_package_data=True,

    license="MIT",

    keywords=['toycv'],

    classifiers=[  # 一些网站的分类信息，方便用户检索
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ])

"""
rm -rf ./dist
pipreqs ./ --encoding=utf8 --force --mode no-pin
python3 setup.py sdist 
twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ -U toycv

twine upload --repository pypi dist/*

pip install .
"""
