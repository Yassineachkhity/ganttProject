document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('scheduler-form');
    const generateTablesBtn = document.getElementById('generate_tables');
    const solveButton = document.getElementById('solve_button');
    const tablesContainer = document.getElementById('tables_container');
    const tablesSuccessMessage = document.getElementById('tables_success_message');

    // Initially disable solve button
    solveButton.disabled = true;

    // Handle table generation
    generateTablesBtn.addEventListener('click', function() {
        const numJobs = parseInt(document.getElementById('num_jobs').value);
        const numMachines = parseInt(document.getElementById('num_machines').value);

        if (!numJobs || !numMachines) {
            alert('Please enter the number of jobs and machines first.');
            return;
        }

        // Show the tables container
        tablesContainer.classList.remove('hidden');
        
        // Show success message
        tablesSuccessMessage.classList.remove('hidden');
        
        // Enable the solve button
        solveButton.disabled = false;

        // Hide success message after 5 seconds
        setTimeout(() => {
            tablesSuccessMessage.classList.add('hidden');
        }, 5000);
    });

    // Validate form inputs
    form.addEventListener('input', function(e) {
        const numJobs = document.getElementById('num_jobs').value;
        const numMachines = document.getElementById('num_machines').value;
        
        // Enable/disable generate tables button based on input validity
        generateTablesBtn.disabled = !numJobs || !numMachines;
    });
});
