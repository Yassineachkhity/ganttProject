document.addEventListener('DOMContentLoaded', function() {
    const numMachinesInput = document.getElementById('num_machines');
    const numJobsInput = document.getElementById('num_jobs');
    const generateFormBtn = document.getElementById('generate-form');
    const jobDetailsContainer = document.getElementById('job-details-container');
    const schedulerForm = document.getElementById('scheduler-form');

    // Generate processing times table when clicking the generate button
    generateFormBtn.addEventListener('click', function() {
        const numMachines = parseInt(numMachinesInput.value);
        const numJobs = parseInt(numJobsInput.value);
        if (numMachines > 0 && numJobs > 0) {
            generateJobDetailsForm(numMachines, numJobs);
        }
    });

    // Handle form submission
    schedulerForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        try {
            // Show loading state
            const resultsDiv = document.getElementById('results') || document.createElement('div');
            resultsDiv.id = 'results';
            resultsDiv.className = 'mt-8 bg-white shadow-lg rounded-lg p-6';
            if (!document.getElementById('results')) {
                schedulerForm.parentNode.appendChild(resultsDiv);
            }
            
            resultsDiv.innerHTML = '<div class="flex justify-center items-center p-4"><div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div></div>';

            // Collect form data
            const formData = {
                rule: document.getElementById('rule').value,
                num_jobs: parseInt(document.getElementById('num_jobs').value),
                num_machines: parseInt(document.getElementById('num_machines').value),
                no_wait: document.getElementById('no_wait').checked,
                no_idle: document.getElementById('no_idle').checked,
                blocking: document.getElementById('blocking').checked,
                processing_times: []
            };

            // Collect processing times
            for (let i = 1; i <= formData.num_jobs; i++) {
                const jobTimes = [];
                for (let j = 1; j <= formData.num_machines; j++) {
                    const input = document.querySelector(`input[data-job="${i}"][data-machine="${j}"]`);
                    if (!input) {
                        throw new Error('Please generate and fill in the processing times table first.');
                    }
                    const value = parseFloat(input.value);
                    if (isNaN(value) || value < 0) {
                        throw new Error(`Invalid processing time for Job ${i}, Machine ${j}`);
                    }
                    jobTimes.push(value);
                }
                formData.processing_times.push(jobTimes);
            }

            // Send data to server
            const response = await fetch('/solve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            // Display results
            resultsDiv.innerHTML = `
                <div id="performance-metrics" class="mb-6">
                    <h3 class="text-xl font-bold mb-4">Performance Metrics</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        <div class="bg-gray-50 p-4 rounded-lg">
                            <p class="font-semibold">Total Time (Makespan)</p>
                            <p class="text-lg">${result.total_time.toFixed(2)}</p>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-lg">
                            <p class="font-semibold">Total Flow Time</p>
                            <p class="text-lg">${result.total_flow_time.toFixed(2)}</p>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-lg">
                            <p class="font-semibold">TFR (Total Flow Ratio)</p>
                            <p class="text-lg">${result.tfr.toFixed(2)}</p>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-lg">
                            <p class="font-semibold">TAR (Time in Average Ratio)</p>
                            <p class="text-lg">${result.tar.toFixed(2)}</p>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-lg">
                            <p class="font-semibold">Cₘₐₓ (Makespan)</p>
                            <p class="text-lg">${result.c_max.toFixed(2)}</p>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-lg">
                            <p class="font-semibold">Job Order</p>
                            <p class="text-lg">Jobs ${result.job_order.join(' → ')}</p>
                        </div>
                    </div>
                </div>
                <div id="gantt-chart">
                    <h3 class="text-xl font-bold mb-4">Gantt Chart</h3>
                    <div class="w-full overflow-x-auto">
                        <img src="data:image/png;base64,${result.gantt_chart}" alt="Gantt Chart" class="max-w-full">
                    </div>
                </div>
            `;

        } catch (error) {
            const resultsDiv = document.getElementById('results');
            if (resultsDiv) {
                resultsDiv.innerHTML = `
                    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                        <strong class="font-bold">Error!</strong>
                        <span class="block sm:inline">${error.message}</span>
                    </div>
                `;
            }
            console.error('Error:', error);
        }
    });
});

// Function to generate the job details form
function generateJobDetailsForm(numMachines, numJobs) {
    const container = document.getElementById('job-details-container');
    container.innerHTML = `
        <h3 class="text-lg font-semibold mb-4">Processing Times</h3>
        <div class="overflow-x-auto">
            <table class="min-w-full bg-white">
                <thead>
                    <tr>
                        <th class="px-4 py-2 border">Job/Machine</th>
                        ${Array.from({length: numMachines}, (_, i) => 
                            `<th class="px-4 py-2 border">Machine ${i + 1}</th>`
                        ).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${Array.from({length: numJobs}, (_, i) => `
                        <tr>
                            <td class="px-4 py-2 border font-semibold">Job ${i + 1}</td>
                            ${Array.from({length: numMachines}, (_, j) => `
                                <td class="px-4 py-2 border">
                                    <input type="number" 
                                           data-job="${i + 1}" 
                                           data-machine="${j + 1}"
                                           class="w-full px-2 py-1 border rounded"
                                           min="0"
                                           step="0.1"
                                           required>
                                </td>
                            `).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}
