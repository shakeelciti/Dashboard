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
        
        grouped = grouped.sort_values(measure, ascending=False)
        
        print(f"Grouped data shape: {grouped.shape}")
        print(f"Grouped data:\n{grouped}")
        
        # Create visualization
        if dimension2_use and use_facet:
            print("Creating grouped chart with colors (facet behavior disabled for better scale)...")
            # Note: We use barmode='group' instead of facet_col because facet_col scales each subplot independently
            # which distorts large values. Using barmode='group' keeps consistent scaling.
            fig = px.bar(grouped,
                        x=dimension1,
                        y=measure,
                        color=dimension2_use,
                        barmode='group',
                        title=f'{measure} by {dimension1} (Grouped by {dimension2_use})')
        elif dimension2_use:
            print("Creating grouped chart...")
            fig = px.bar(grouped,
                        x=dimension1,
                        y=measure,
                        color=dimension2_use,
                        barmode='group',
                        title=f'{measure} by {dimension1} (Grouped by {dimension2_use})')
        else:
            print("Creating simple bar chart...")
            fig = px.bar(grouped,
                        x=dimension1,
                        y=measure,
                        title=f'{measure} by {dimension1}')
        
        # Update layout
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            template='plotly_white',
            hovermode='x unified'
        )
        
        print("Chart created successfully")
        return jsonify({'chart': fig.to_json()})
        
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
        
        # Clean data
        df_clean = df[[dimension, measure]].dropna()
        
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
        print(f"Grouped data:\n{grouped}")
        
        # Limit to top 10
        if len(grouped) > 10:
            grouped = grouped.nlargest(10, measure)
        
        # Create chart
        fig = px.pie(grouped, names=dimension, values=measure,
                    title=f'{measure} Distribution by {dimension}')
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
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
        
        # Clean data
        df_clean = df[dims + [measure]].dropna()
        
        if df_clean.empty:
            return jsonify({'error': 'No complete data available'}), 400
        
        # Convert measure to numeric
        print(f"Measure dtype before: {df_clean[measure].dtype}")
        df_clean[measure] = pd.to_numeric(df_clean[measure], errors='coerce')
        df_clean = df_clean.dropna(subset=[measure])
        
        print(f"Measure dtype after: {df_clean[measure].dtype}")
        
        if not pd.api.types.is_numeric_dtype(df_clean[measure]):
            return jsonify({'error': f'{measure} is not numeric'}), 400
        
        # Convert to string
        for dim in dims:
            df_clean[dim] = df_clean[dim].astype(str)
        
        # Filter positive values
        df_clean = df_clean[df_clean[measure] > 0]
        
        if df_clean.empty:
            return jsonify({'error': 'No positive values found'}), 400
        
        print(f"Cleaned data shape: {df_clean.shape}")
        print(f"Cleaned data:\n{df_clean}")
        
        # Aggregate data by all three dimensions
        grouped = df_clean.groupby(dims, as_index=False)[measure].sum()
        print(f"\nGrouped data shape: {grouped.shape}")
        print(f"Grouped data:\n{grouped}")
        
        # Limit data
        if len(grouped) > 1000:
            grouped = grouped.nlargest(1000, measure)
            print(f"Data limited to top 1000 rows")
        
        # Create chart
        fig = px.sunburst(grouped, path=dims, values=measure,
                         title=f'{measure}: {dim1} → {dim2} → {dim3}')
        
        fig.update_traces(textinfo='label+percent entry')
        fig.update_layout(height=650)
        
        print("Sunburst chart created successfully")
        return jsonify({'chart': fig.to_json()})
        
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