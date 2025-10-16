import json
from typing import Any


class PlotTool:
    """Tool for creating 2D plots that can be embedded in the chat"""

    @staticmethod
    def get_definition() -> dict:
        """
        Return the tool definition for the LLM in OpenAI function calling format.
        This tells the LLM exactly how to use this tool.
        """
        return {
            "type": "function",
            "function": {
                "name": "create_plot",
                "description": """Creates a 2D plot visualization that will be displayed directly in the chat.

Use this tool when the user asks to visualize data, create a chart, plot graphs, or show any 2D data visualization.

The plot supports multiple chart types:
- line: Line charts for continuous data, trends over time
- bar: Bar charts for comparing categories or discrete values
- scatter: Scatter plots for showing relationships between two variables
- area: Area charts for showing cumulative data or filled line charts

You can add multiple datasets to the same plot for comparison.

IMPORTANT: Provide ALL the data points needed for the plot. Don't reference external data - include the actual values.

Examples:
1. "Plot the function y=x^2 from -10 to 10" -> Generate x values from -10 to 10, calculate y values, use line chart
2. "Show sales data for Q1: Jan=100, Feb=150, Mar=120" -> Use bar chart with these exact values
3. "Compare temperature in NYC vs LA over a week" -> Use line chart with two datasets""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the plot that describes what is being visualized"
                        },
                        "x_label": {
                            "type": "string",
                            "description": "Label for the x-axis (horizontal axis)"
                        },
                        "y_label": {
                            "type": "string",
                            "description": "Label for the y-axis (vertical axis)"
                        },
                        "datasets": {
                            "type": "array",
                            "description": "Array of datasets to plot. Each dataset represents one line/series on the plot.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "Name/legend label for this dataset (e.g., 'Temperature', 'Sales', 'Dataset 1')"
                                    },
                                    "x_values": {
                                        "type": "array",
                                        "description": "Array of x-axis values (numbers or strings for categories)",
                                        "items": {
                                            "oneOf": [
                                                {"type": "number"},
                                                {"type": "string"}
                                            ]
                                        }
                                    },
                                    "y_values": {
                                        "type": "array",
                                        "description": "Array of y-axis values (must be numbers, same length as x_values)",
                                        "items": {
                                            "type": "number"
                                        }
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": ["line", "bar", "scatter", "area"],
                                        "description": "Type of chart for this dataset. Default is 'line'."
                                    },
                                    "color": {
                                        "type": "string",
                                        "description": "Optional hex color code (e.g., '#FF5733'). Will auto-generate if not provided."
                                    }
                                },
                                "required": ["label", "x_values", "y_values"]
                            }
                        }
                    },
                    "required": ["title", "x_label", "y_label", "datasets"]
                }
            }
        }

    @staticmethod
    def execute(
        title: str,
        x_label: str,
        y_label: str,
        datasets: list[dict]
    ) -> str:
        """
        Execute the plot tool and return HTML/JS visualization

        Returns: HTML string with embedded Chart.js visualization
        """
        # Generate unique ID for this chart
        import uuid
        chart_id = f"chart_{uuid.uuid4().hex[:8]}"

        # Default colors for datasets
        default_colors = [
            '#3b82f6',  # blue
            '#ef4444',  # red
            '#10b981',  # green
            '#f59e0b',  # orange
            '#8b5cf6',  # purple
            '#ec4899',  # pink
            '#14b8a6',  # teal
            '#f97316',  # orange-red
        ]

        # Prepare datasets for Chart.js
        chart_datasets = []
        for i, dataset in enumerate(datasets):
            color = dataset.get('color', default_colors[i % len(default_colors)])
            chart_type = dataset.get('type', 'line')

            dataset_config = {
                'label': dataset['label'],
                'data': [
                    {'x': x, 'y': y}
                    for x, y in zip(dataset['x_values'], dataset['y_values'])
                ],
                'borderColor': color,
                'backgroundColor': color + '33',  # Add transparency
                'borderWidth': 2,
                'tension': 0.1,
                'type': chart_type
            }

            # Special config for scatter plots
            if chart_type == 'scatter':
                dataset_config['pointRadius'] = 5
                dataset_config['pointHoverRadius'] = 7

            # Special config for bar charts
            if chart_type == 'bar':
                dataset_config['backgroundColor'] = color
                dataset_config['borderColor'] = color
                dataset_config['borderWidth'] = 1

            # Special config for area charts
            if chart_type == 'area':
                dataset_config['fill'] = True
                dataset_config['type'] = 'line'

            chart_datasets.append(dataset_config)

        # Generate HTML with Chart.js
        # Note: Chart.js is already loaded in the main page
        html = f"""
        <div style="background: white; padding: 20px; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); width: 100%;">
            <canvas id="{chart_id}" style="width: 100%; height: 400px;"></canvas>
        </div>
        <script>
        (function() {{
            const ctx = document.getElementById('{chart_id}');
            if (!ctx) {{
                console.error('Chart canvas not found: {chart_id}');
                return;
            }}
            if (typeof Chart === 'undefined') {{
                console.error('Chart.js not loaded yet');
                return;
            }}

            new Chart(ctx, {{
                data: {{
                    datasets: {json.dumps(chart_datasets)}
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: {json.dumps(title)},
                            font: {{
                                size: 16,
                                weight: 'bold'
                            }}
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'linear',
                            title: {{
                                display: true,
                                text: {json.dumps(x_label)}
                            }},
                            ticks: {{
                                callback: function(value) {{
                                    return value;
                                }}
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: {json.dumps(y_label)}
                            }},
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        }})();
        </script>
        """

        return html


class ToolRegistry:
    """Registry of all available tools"""

    def __init__(self):
        self.tools = {
            "create_plot": PlotTool()
        }

    def get_tool_definitions(self) -> list[dict]:
        """Get all tool definitions for the LLM"""
        return [
            PlotTool.get_definition()
        ]

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool by name with given arguments"""
        if tool_name == "create_plot":
            return PlotTool.execute(**arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
