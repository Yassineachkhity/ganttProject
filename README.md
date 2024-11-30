# Scheduling Solver Project

## Overview
This is a web-based scheduling solver application that helps solve various scheduling problems using Mixed Integer Linear Programming (MILP).

## Features
- Support for different shop types:
  - Flow Shop
  - Job Shop
  - Open Shop
- Constraint selection
- Multiple optimization criteria
- Dynamic input for processing and setup times

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository
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
flask run
```

## Usage
1. Select the shop type (Flow Shop, Job Shop, Open Shop)
2. Set the number of machines and jobs
3. Input processing and setup times
4. Select constraints and optimization criteria
5. Click "Solve Scheduling Problem"

## Constraints Supported
- No Idle Time
- No Wait
- Machine Blocking
- Sequence Dependent Setup Times

## Optimization Criteria
- Minimize Makespan
- Minimize Total Flow Time
- Minimize Maximum Tardiness

## Technologies
- Flask
- PuLP (Linear Programming)
- Bootstrap
- JavaScript

## Contributing
Contributions are welcome! Please submit pull requests or open issues.

## License
MIT License
