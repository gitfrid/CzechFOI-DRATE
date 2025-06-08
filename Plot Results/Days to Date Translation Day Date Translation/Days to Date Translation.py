import plotly.graph_objects as go
import pandas as pd
import os
import shutil

def main():
    
    plot_name = "Day Date Translation"

    # Initialize the directory and script copy
    full_plotfile_name = init_function(plot_name)

    # Define the start date
    start_date = pd.to_datetime('2020-01-01')

    # Generate the day numbers (0 to 1533)
    day_numbers = list(range(0, 1534))

    # Calculate the corresponding dates for each day number
    dates = [start_date + pd.Timedelta(days=d) for d in day_numbers]

    # Select the first of each month
    monthly_dates = [date for date in dates if date.day == 1]

    # Get the corresponding day numbers for the first of each month
    monthly_day_numbers = [day_numbers[dates.index(date)] for date in monthly_dates]

    # Create the Plotly graph
    fig = go.Figure()

    # Plot the dates as scatter points
    fig.add_trace(go.Scatter(
        x=monthly_dates,  # dates on the x-axis
        y=[1]*len(monthly_dates),  # Dummy y-values since we only need the x-axis
        mode='markers',  # Only markers (dots)
        marker=dict(size=1, color='light grey', symbol='circle')
    ))

    # Add annotations for day numbers above each date (moving them closer to the dates)
    annotations = []
    for i, date in enumerate(monthly_dates):
        annotations.append(dict(
            x=date,
            y=0.98,  # Lower the y value to bring the day numbers closer to the dates (negative offset)
            text=str(monthly_day_numbers[i]),
            showarrow=False,
            font=dict(size=12, color='black'),
            textangle=90,  # Rotate text vertically (90 degrees)
            xanchor='center',
            yanchor='bottom'
        ))

    # Update layout for the x-axis and y-axis
    fig.update_layout(
        title="Day Numbers and Corresponding First of Each Month (0-1533)",
        xaxis_title="Date",
        yaxis_title="",  # No y-axis label
        xaxis=dict(
            tickmode='array',
            tickvals=monthly_dates,  # Use the dates for the x-axis ticks
            ticktext=[d.strftime('%Y-%m-%d') for d in monthly_dates],  # Show the formatted dates as tick labels
            tickangle=45,  # Rotate the date labels for better readability
            ticklen=10,  # Add length to the ticks for clarity
        ),
        yaxis=dict(
            showticklabels=False,  # Hide y-axis labels
            zeroline=False,  # Remove the zero line
            showline=False,  # Hide the y-axis line
        ),
        showlegend=False,
        annotations=annotations  # Add annotations for day numbers
    )

    #fig.show()
    fig.write_html(f"{full_plotfile_name} .html")
    print(f"Plot has been saved to HTML file.")


# Initialize the "Plot Results" directory and copy this PyScript into it
def init_function(plot_name):
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    first_root_dir = os.path.abspath(os.sep.join(script_dir.split(os.sep)[:2]))
    plot_result_path = os.path.join(first_root_dir, "Plot Results", f"{script_name} {plot_name}")
    os.makedirs(plot_result_path, exist_ok=True)
    
    script_file_name = os.path.join(plot_result_path, f"{script_name}.py")
    if not os.path.exists(script_file_name):
        shutil.copy(__file__, script_file_name)
    # return plot file name without extension
    return plot_result_path + os.sep + f"{script_name} {plot_name}"

# call main script
if __name__ == "__main__": main()
