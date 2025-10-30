@echo off
echo Setting up hierarchical-ei project structure...

:: Create all directories
mkdir heei\core 2>nul
mkdir heei\metrics 2>nul
mkdir heei\models 2>nul
mkdir heei\data 2>nul
mkdir heei\utils 2>nul
mkdir experiments\configs 2>nul
mkdir experiments\results 2>nul
mkdir notebooks 2>nul
mkdir scripts 2>nul
mkdir tests 2>nul
mkdir docs 2>nul
mkdir supplementary\figures 2>nul

:: Create files
type nul > README.md
type nul > requirements.txt
type nul > setup.py
type nul > .gitignore
type nul > heei\__init__.py
type nul > heei\core\__init__.py
type nul > heei\metrics\__init__.py
type nul > heei\models\__init__.py
type nul > heei\data\__init__.py
type nul > heei\utils\__init__.py

echo Project structure created successfully!
pause 