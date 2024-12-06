# Gantt Project Solver

## Overview
This is a web application developed by ENSAM Casablanca students that helps solve and visualize Gantt chart scheduling problems using Mixed Integer Linear Programming (MILP). The project aims to provide an intuitive interface for solving complex scheduling problems in manufacturing and project management.

## Features
- Interactive web interface for problem input
- Support for multiple scheduling scenarios:
  - Flow Shop Scheduling
  - Job Shop Scheduling
  - Open Shop Scheduling
- Real-time solution visualization
- Constraint handling and optimization
- Responsive design for all devices

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository
```bash
git clone https://github.com/Yassineachkhity/ganttProject.git
cd ganttProject
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app.py
```

## Running the Application Locally

1. Create a virtual environment:
```bash
python -m venv .venv
```
2. Activate the virtual environment:
- Windows:
```bash
.venv\Scripts\activate
```
- Unix or MacOS:
```bash
source .venv/bin/activate
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Run the Flask application:
```bash
python app.py
```
The application will be available at `http://127.0.0.1:5000/`

## Usage
1. Navigate to the home page
2. Choose your scheduling problem type
3. Input your problem parameters:
   - Number of machines
   - Number of jobs
   - Processing times
   - Constraints
4. Click "Solve" to generate the solution
5. View the resulting Gantt chart

## Technologies Used
- Backend:
  - Flask (Python web framework)
  - PuLP (Optimization library)
- Frontend:
  - HTML5/CSS3
  - JavaScript
  - Bootstrap 5
- Version Control:
  - Git

## Team Members
- ENSAM CASABLANCA Students

## Contributing
We welcome contributions from the community. Please feel free to submit issues and pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
