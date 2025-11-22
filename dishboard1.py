from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from werkzeug.utils import secure_filename
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

app.secret_key = 'your-secret-key-here-change-this'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Favicon route to eliminate 404 errors
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return empty response with 204 No Content status

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_data(filepath):
    """Load Excel or CSV file"""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        return df, None
    except Exception as e:
        return None, str(e)

def analyze_columns(df):
    """Analyze and categorize columns"""
    # Start with basic type detection
    dimensions = df.select_dtypes(include=['object', 'category']).columns.tolist()
    measures = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    # Try to convert object columns to numeric and identify actual measures
    for col in dimensions[:]:  # Create a copy to iterate over
        try:
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            # If most values can be converted to numeric, move to measures
            if numeric_vals.notna().sum() / len(df) > 0.8:  # At least 80% numeric
                dimensions.remove(col)
                measures.append(col)
        except:
            pass

    # Remove ID-like columns from measures
    id_keywords = ['id', 'ID', 'Id', 'code', 'Code', 'number', 'Number', 'Code']
    measures = [col for col in measures if not any(keyword in col for keyword in id_keywords)]

    # Remove columns with too many unique values (likely IDs)
    measures = [col for col in measures if df[col].nunique() < len(df) * 0.95]

    # Remove measures with no variation
    measures = [col for col in measures if df[col].nunique() > 1]

    # Remove columns that are all NaN
    measures = [col for col in measures if df[col].notna().sum() > 0]

    return dimensions, measures

@app.route('/')
def index():
    """Home page with file upload"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload Excel (.xlsx, .xls) or CSV files'}), 400

    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and analyze data
        df, error = load_data(filepath)
        if error:
            return jsonify({'error': f'Error loading file: {error}'}), 400

        dimensions, measures = analyze_columns(df)

        # Store filepath in session
        session['filepath'] = filepath
        session['filename'] = filename

        return jsonify({
            'success': True,
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'dimensions': dimensions,
            'measures': measures,
            'redirect': '/dashboard'
        })

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/dashboard')
def dashboard():
    """Dashboard page with visualizations"""
    filepath = session.get('filepath')
    filename = session.get('filename')

    if not filepath or not os.path.exists(filepath):
        return render_template('index.html', error='Please upload a file first')

    df, error = load_data(filepath)
    if error:
        return render_template('index.html', error=f'Error loading data: {error}')

    dimensions, measures = analyze_columns(df)

    return render_template('dashboard.html',
                         filename=filename,
                         rows=len(df),
                         columns=len(df.columns),
                         dimensions=dimensions,
                         measures=measures)

@app.route('/api/chart/bar', methods=['POST'])
def create_bar_chart():
    """API endpoint to create bar chart with optional second dimension and grouping"""
    try:
        import traceback
        print("\n=== BAR CHART REQUEST RECEIVED ===")

        data = request.json
        print(f"Request data: {data}")

        dimension1 = data.get('dimension1')
        dimension2 = data.get('dimension2')
        measure = data.get('measure')
        use_facet = data.get('use_facet', False)

        print(f"D1: {dimension1}, D2: {dimension2}, M: {measure}, Facet: {use_facet}")

        # Validate inputs
        if not dimension1 or not measure:
            return jsonify({'error': 'dimension1 and measure are required'}), 400

        filepath = session.get('filepath')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'No data loaded'}), 400

        # Load data
        df, error = load_data(filepath)
        if error:
            return jsonify({'error': f'Error loading data: {error}'}), 400

        print(f"Data loaded. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Prepare data columns
        cols_to_use = [dimension1, measure]
        if dimension2 and dimension2 != '' and dimension2 != 'None':
            cols_to_use.append(dimension2)
            dimension2_use = dimension2
        else:
            dimension2_use = None

        # Verify columns exist
        for col in cols_to_use:
            if col not in df.columns:
                return jsonify({'error': f'Column "{col}" not found in data'}), 400

        # Select and clean data
        df_work = df[cols_to_use].copy()
        print(f"Before dropna: {df_work.shape}")
        df_work = df_work.dropna()
        print(f"After dropna: {df_work.shape}")

        if df_work.empty:
            return jsonify({'error': 'No valid data after removing nulls'}), 400

        # Convert measure to numeric
        print(f"Measure column dtype before: {df_work[measure].dtype}")
        print(f"Sample values: {df_work[measure].head(10).tolist()}")

        df_work[measure] = pd.to_numeric(df_work[measure], errors='coerce')
        df_work = df_work.dropna(subset=[measure])

        print(f"Measure column dtype after: {df_work[measure].dtype}")
        print(f"Sample values after conversion: {df_work[measure].head(10).tolist()}")

        if df_work.empty:
            return jsonify({'error': 'No numeric values in measure column'}), 400

        # Aggregate data
        if dimension2_use:
            grouped = df_work.groupby([dimension1, dimension2_use], as_index=False)[measure].sum()
        else:
            grouped = df_work.groupby(dimension1, as_index=False)[measure].sum()

        grouped = grouped.sort_values(measure, ascending=False).reset_index(drop=True)

        print(f"Grouped data shape: {grouped.shape}")
        print(f"Grouped data:\n{grouped}")
        print(f"Grouped data dtypes:\n{grouped.dtypes}")
        print(f"Grouped data values: {grouped[measure].tolist()}")

        # Create visualization - use go.Bar for more explicit control
        if dimension2_use:
            print("Creating grouped chart with dimension2...")
            # Get unique values for dimension2
            categories = grouped[dimension2_use].unique()
            fig = go.Figure()
            
            for category in categories:
                cat_data = grouped[grouped[dimension2_use] == category]
                fig.add_trace(go.Bar(
                    x=cat_data[dimension1],
                    y=cat_data[measure].astype(float).tolist(),
                    name=str(category),
                    text=cat_data[measure].astype(float).apply(lambda x: f'{x:.2f}').tolist(),
                    textposition='outside'
                ))
            
            fig.update_layout(
                barmode='group',
                title=f'{measure} by {dimension1} (Grouped by {dimension2_use})',
                xaxis_title=dimension1,
                yaxis_title=measure,
                height=500,
                template='plotly_white',
                hovermode='x unified'
            )
        else:
            print("Creating simple bar chart...")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=grouped[dimension1],
                y=grouped[measure].astype(float).tolist(),
                text=grouped[measure].astype(float).apply(lambda x: f'{x:.2f}').tolist(),
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f'{measure} by {dimension1}',
                xaxis_title=dimension1,
                yaxis_title=measure,
                height=500,
                xaxis_tickangle=-45,
                template='plotly_white',
                hovermode='x unified'
            )

        print("Chart created successfully")
        print(f"Chart traces: {fig.data}")
        print(f"Chart layout: {fig.layout}")
        
        chart_json = fig.to_json()
        print(f"Chart JSON length: {len(chart_json)}")
        
        return jsonify({'chart': chart_json})

    except Exception as e:
        error_msg = str(e)
        print(f"\n!!! ERROR in bar chart: {error_msg}")
        print(traceback.format_exc())
        print("=== END ERROR ===\n")
        return jsonify({'error': error_msg}), 500
    
@app.route('/api/chart/pie', methods=['POST'])
def create_pie_chart():
    """API endpoint to create pie chart"""
    try:
        import traceback
        print("\n=== PIE CHART REQUEST RECEIVED ===")
        data = request.json
        dimension = data.get('dimension')
        measure = data.get('measure')
        print(f"Dimension: {dimension}, Measure: {measure}")
        
        filepath = session.get('filepath')
        if not filepath:
            return jsonify({'error': 'No data loaded'}), 400
        
        df, error = load_data(filepath)
        if error:
            return jsonify({'error': error}), 400
        
        print(f"Data loaded. Shape: {df.shape}")
        
        # Clean data - Create a copy first
        df_clean = df[[dimension, measure]].copy()
        df_clean = df_clean.dropna()
        
        if df_clean.empty:
            return jsonify({'error': 'No valid data'}), 400
        
        # Convert measure to numeric
        print(f"Measure dtype before: {df_clean[measure].dtype}")
        df_clean[measure] = pd.to_numeric(df_clean[measure], errors='coerce')
        df_clean = df_clean.dropna(subset=[measure])
        
        print(f"Measure dtype after: {df_clean[measure].dtype}")
        
        if not pd.api.types.is_numeric_dtype(df_clean[measure]):
            return jsonify({'error': f'{measure} is not numeric'}), 400
        
        # Group and filter
        grouped = df_clean.groupby(dimension)[measure].sum().reset_index()
        grouped = grouped[grouped[measure] > 0]
        
        print(f"Grouped data shape: {grouped.shape}")
        print(f"Grouped data:\n{grouped.to_string()}")
        
        # Limit to top 10 and sort
        if len(grouped) > 10:
            grouped = grouped.nlargest(10, measure)
        
        grouped = grouped.sort_values(measure, ascending=False)
        
        print(f"Final data for chart:\n{grouped.to_string()}")
        
        # **KEY FIX**: Create chart using lists instead of dataframe
        labels = grouped[dimension].tolist()
        values = grouped[measure].tolist()
        
        # Debug print
        print(f"Labels: {labels}")
        print(f"Values: {values}")
        print(f"Values type: {[type(v) for v in values]}")
        
        # Create chart with explicit data
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            textposition='auto',
            textinfo='percent+label+value',
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.2f}<br>Percent: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'{measure} Distribution by {dimension}',
            height=500
        )
        
        # Debug: Check the actual figure data
        chart_dict = fig.to_dict()
        print(f"Chart values from dict: {chart_dict['data'][0]['values']}")
        
        print("Pie chart created successfully")
        return jsonify({'chart': fig.to_json()})
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n!!! ERROR in pie chart: {error_msg}")
        print(traceback.format_exc())
        print("=== END ERROR ===\n")
        return jsonify({'error': error_msg}), 500

@app.route('/api/chart/sunburst', methods=['POST'])
def create_sunburst_chart():
    """API endpoint to create sunburst chart"""
    try:
        import traceback
        print("\n=== SUNBURST CHART REQUEST RECEIVED ===")

        data = request.json
        dim1 = data.get('dim1')
        dim2 = data.get('dim2')
        dim3 = data.get('dim3')
        measure = data.get('measure')

        print(f"D1: {dim1}, D2: {dim2}, D3: {dim3}, M: {measure}")

        dims = [dim1, dim2, dim3]

        # Validate unique dimensions
        if len(set(dims)) < len(dims):
            return jsonify({'error': 'All three dimensions must be different'}), 400

        filepath = session.get('filepath')
        if not filepath:
            return jsonify({'error': 'No data loaded'}), 400

        df, error = load_data(filepath)
        if error:
            return jsonify({'error': error}), 400

        print(f"Data loaded. Shape: {df.shape}")

        # Clean data - Create a copy first
        df_clean = df[dims + [measure]].copy()
        df_clean = df_clean.dropna()

        if df_clean.empty:
            return jsonify({'error': 'No complete data available'}), 400

        # Convert measure to numeric
        print(f"Measure dtype before: {df_clean[measure].dtype}")
        df_clean[measure] = pd.to_numeric(df_clean[measure], errors='coerce')
        df_clean = df_clean.dropna(subset=[measure])

        print(f"Measure dtype after: {df_clean[measure].dtype}")

        if not pd.api.types.is_numeric_dtype(df_clean[measure]):
            return jsonify({'error': f'{measure} is not numeric'}), 400

        # Convert dimensions to string
        for dim in dims:
            df_clean[dim] = df_clean[dim].astype(str)

        # Filter positive values
        df_clean = df_clean[df_clean[measure] > 0]

        if df_clean.empty:
            return jsonify({'error': 'No positive values found'}), 400

        print(f"Cleaned data shape: {df_clean.shape}")

        # Aggregate data by all three dimensions
        grouped = df_clean.groupby(dims, as_index=False)[measure].sum()
        
        # Ensure measure is float type
        grouped[measure] = grouped[measure].astype(float)
        
        print(f"\nGrouped data shape: {grouped.shape}")
        print(f"Grouped data:\n{grouped.to_string()}")

        # Limit data
        if len(grouped) > 500:
            grouped = grouped.nlargest(500, measure)
            print(f"Data limited to top 500 rows")

        # Build hierarchical structure for sunburst
        labels = []
        parents = []
        values = []
        ids = []  # Add unique IDs to avoid conflicts
        
        # Add root
        labels.append("Total")
        parents.append("")
        values.append(float(grouped[measure].sum()))
        ids.append("root")
        
        # Level 1: dim1
        level1_agg = grouped.groupby(dim1)[measure].sum()
        for cat1 in level1_agg.index:
            cat1_str = str(cat1)
            labels.append(cat1_str)
            parents.append("root")
            values.append(float(level1_agg[cat1]))
            ids.append(f"L1_{cat1_str}")
        
        # Level 2: dim1 -> dim2
        level2_agg = grouped.groupby([dim1, dim2])[measure].sum()
        for (cat1, cat2) in level2_agg.index:
            cat1_str = str(cat1)
            cat2_str = str(cat2)
            labels.append(cat2_str)
            parents.append(f"L1_{cat1_str}")
            values.append(float(level2_agg[(cat1, cat2)]))
            ids.append(f"L2_{cat1_str}_{cat2_str}")
        
        # Level 3: dim1 -> dim2 -> dim3 (leaf nodes)
        for idx, row in grouped.iterrows():
            cat1_str = str(row[dim1])
            cat2_str = str(row[dim2])
            cat3_str = str(row[dim3])
            val = float(row[measure])
            
            labels.append(cat3_str)
            parents.append(f"L2_{cat1_str}_{cat2_str}")
            values.append(val)
            ids.append(f"L3_{cat1_str}_{cat2_str}_{cat3_str}_{idx}")
        
        # Debug output
        print(f"\nSunburst structure:")
        print(f"Total nodes: {len(labels)}")
        print(f"First 5 labels: {labels[:5]}")
        print(f"First 5 parents: {parents[:5]}")
        print(f"First 5 values: {values[:5]}")
        print(f"Sum of all values: {sum(values)}")
        
        # Validate structure
        if len(labels) != len(parents) or len(labels) != len(values) or len(labels) != len(ids):
            error_msg = f"Data structure mismatch: labels={len(labels)}, parents={len(parents)}, values={len(values)}, ids={len(ids)}"
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            textinfo='label+percent entry',
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.2f}<br>%{percentParent} of parent<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{measure}: {dim1} → {dim2} → {dim3}',
            height=650,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        # Convert to JSON
        chart_json = fig.to_json()
        
        # Verify JSON is valid
        chart_dict = json.loads(chart_json)
        print(f"Chart data validation:")
        print(f"  - Has 'data' key: {'data' in chart_dict}")
        print(f"  - Has 'layout' key: {'layout' in chart_dict}")
        print(f"  - Number of data traces: {len(chart_dict.get('data', []))}")
        if chart_dict.get('data'):
            print(f"  - First trace type: {chart_dict['data'][0].get('type')}")
            print(f"  - First trace has values: {'values' in chart_dict['data'][0]}")
            print(f"  - Number of values: {len(chart_dict['data'][0].get('values', []))}")

        print("Sunburst chart created successfully\n")
        return jsonify({'chart': chart_json})

    except Exception as e:
        error_msg = str(e)
        print(f"\n!!! ERROR in sunburst chart: {error_msg}")
        print(traceback.format_exc())
        print("=== END ERROR ===\n")
        return jsonify({'error': error_msg}), 500

@app.route('/api/data-info')
def get_data_info():
    """Get information about loaded data"""
    filepath = session.get('filepath')
    if not filepath:
        return jsonify({'error': 'No data loaded'}), 400

    df, error = load_data(filepath)
    if error:
        return jsonify({'error': error}), 400

    dimensions, measures = analyze_columns(df)

    return jsonify({
        'rows': len(df),
        'columns': len(df.columns),
        'dimensions': dimensions,
        'measures': measures,
        'sample': df.head(5).to_dict('records')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)