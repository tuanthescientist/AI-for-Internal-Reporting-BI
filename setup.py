from setuptools import setup, find_packages

setup(
    name="ai-internal-reporting-bi",
    version="1.0.0",
    author="Tuan Tran",
    description="AI-powered Internal Reporting & Business Intelligence Platform",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tuanthescientist/AI-for-Internal-Reporting-BI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "groq>=0.9.0",
        "PyQt5>=5.15.9",
        "matplotlib>=3.7.1",
        "pandas>=2.0.0",
        "numpy>=1.24.3",
        "python-dotenv>=1.0.0",
        "scipy>=1.10.1",
        "openpyxl>=3.1.2",
        "reportlab>=4.0.4",
        "pydantic>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Accounting",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "bi-platform=run:launch",
        ],
    },
)
