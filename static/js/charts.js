// Chart.js configuration for electricity theft detection

// Colors for classifications
const classificationColors = {
    'normal': '#28a745',
    'suspicious': '#ffc107',
    'theft': '#dc3545',
    'unknown': '#6c757d'
};

// Create consumption timeline chart
function createConsumptionChart(canvasId, chartData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.dates,
            datasets: [
                {
                    label: 'Total Consumption (kWh)',
                    data: chartData.consumptions,
                    backgroundColor: chartData.classification_colors || 
                                   Array(chartData.dates.length).fill(classificationColors.unknown),
                    borderColor: '#495057',
                    borderWidth: 1,
                    yAxisID: 'y',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Consumption (kWh)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const index = context.dataIndex;
                            const consumption = context.parsed.y;
                            const classification = chartData.classifications?.[index] || 'unknown';
                            return [
                                `Consumption: ${consumption.toFixed(2)} kWh`,
                                `Classification: ${classification}`
                            ];
                        }
                    }
                }
            }
        }
    });
}

// Create anomaly score chart
function createAnomalyChart(canvasId, chartData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: [
                {
                    label: 'Anomaly Score',
                    data: chartData.anomaly_scores,
                    borderColor: '#6f42c1',
                    backgroundColor: 'rgba(111, 66, 193, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                },
                {
                    label: 'Normal Threshold',
                    data: Array(chartData.dates.length).fill(chartData.threshold_normal || 0),
                    borderColor: classificationColors.normal,
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                },
                {
                    label: 'Theft Threshold',
                    data: Array(chartData.dates.length).fill(chartData.threshold_theft || -0.1),
                    borderColor: classificationColors.theft,
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Anomaly Score'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                annotation: {
                    annotations: {
                        normalZone: {
                            type: 'box',
                            yMin: chartData.threshold_normal || 0,
                            yMax: 1,
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            borderWidth: 0,
                        },
                        suspiciousZone: {
                            type: 'box',
                            yMin: chartData.threshold_theft || -0.1,
                            yMax: chartData.threshold_normal || 0,
                            backgroundColor: 'rgba(255, 193, 7, 0.1)',
                            borderWidth: 0,
                        },
                        theftZone: {
                            type: 'box',
                            yMin: -1,
                            yMax: chartData.threshold_theft || -0.1,
                            backgroundColor: 'rgba(220, 53, 69, 0.1)',
                            borderWidth: 0,
                        }
                    }
                }
            }
        }
    });
}

// Create classification distribution chart
function createDistributionChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Suspicious', 'Theft'],
            datasets: [{
                data: [data.normal || 0, data.suspicious || 0, data.theft || 0],
                backgroundColor: [
                    classificationColors.normal,
                    classificationColors.suspicious,
                    classificationColors.theft
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((context.parsed / total) * 100);
                            return `${context.label}: ${context.parsed} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Create performance trend chart
function createPerformanceChart(canvasId, performanceData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: performanceData.dates,
            datasets: [
                {
                    label: 'F1 Score',
                    data: performanceData.f1_scores,
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y',
                },
                {
                    label: 'Accuracy',
                    data: performanceData.accuracy_scores,
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Score'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                annotation: {
                    annotations: {
                        threshold: {
                            type: 'line',
                            yMin: 0.5,
                            yMax: 0.5,
                            borderColor: '#dc3545',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: {
                                content: 'Threshold: 0.5',
                                enabled: true,
                                position: 'end'
                            }
                        }
                    }
                }
            }
        }
    });
}

// Update chart with new data
function updateChart(chart, newData) {
    chart.data.labels = newData.dates;
    chart.data.datasets[0].data = newData.values;
    chart.update();
}

// Export chart as image
function exportChart(chart, filename = 'chart.png') {
    const link = document.createElement('a');
    link.download = filename;
    link.href = chart.toBase64Image();
    link.click();
}