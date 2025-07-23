     Patent Analysis Tool
    A comprehensive web application for analyzing patent documents with user authentication and admin dashboard functionality.
    📋 Table of Contents

    Features
    Prerequisites
    Installation
    Running the Application
    Usage
    Admin Access
   

    ✨ Features
    
    User Authentication: Secure sign-up and login system
    Patent Analysis: Upload and analyze patent documents
    Results Management: View and download analysis results
    Admin Dashboard: Administrative interface for system management
    User-Friendly Interface: Intuitive web-based interface
    
    🔧 Prerequisites
    Before running the application, ensure you have the following installed:
    
    Python 3.7 or higher
    Git (optional, for cloning)
    A modern web browser
    
    📥 Installation
    Step 1: Download the Source Code
    
    Option A: Download ZIP
    
    Navigate to: https://github.com/CAYSUS-DILAN/Patent-Analysis-Tool-
    Click the green "Code" button
    Select "Download ZIP"
    Extract the downloaded ZIP file to your desired directory

    
    Option B: Clone Repository (if you have Git installed)
    bashgit clone https://github.com/CAYSUS-DILAN/Patent-Analysis-Tool-.git
    
    
    Step 2: Install IDE (Choose One)
    PyCharm Community Edition (Recommended)
    
    Download from: https://www.jetbrains.com/pycharm/download/?section=windows
    Follow the installation wizard
    
    Visual Studio Code
    
    Download from: https://code.visualstudio.com/docs/setup/windows
    Install Python extension for better support
    
    Step 3: Install Dependencies
    Navigate to the project directory and install required packages:
    bash
    cd Patent-Analysis-Tool-
    pip install -r requirements.txt
    Note: If requirements.txt is not available, install common dependencies:
    bash
    pip install flask flask-sqlalchemy flask-login werkzeug
    🚀 Running the Application
    
    Navigate to Backend Directory
    bashcd backend
    
    Run the Application
    bashpython app.py
    
    Access the Application
    
    The application will start and display a local URL (typically http://127.0.0.1:5000 or http://localhost:5000)
    Copy the generated link/IP address
    Paste it into your web browser
    
    
    
    💻 Usage
    For Regular Users
    
    Sign Up
    
    Navigate to the application URL
    Create a new account using your email and password
    Complete the registration process
    
    
    Login
    
    Use your registered credentials to log in
    Access the main dashboard
    
    
    Upload and Analyze Patents
    
    Upload your patent documents
    Wait for the analysis to complete
    View results on the platform
    
    
    Download Results
    
    Access your analysis results
    Download reports in available formats
    Manage your previous analyses
    
    
    Logout
    
    Safely logout when finished
    
    
    
    User Workflow
    Sign Up → Login → Upload File → View Results → Download Results → Logout

    👨‍💼 Admin Access
    Accessing Admin Dashboard
    
    Start the Application
    
    Follow the running instructions above
    Ensure the app is running successfully
    
    
    Access Admin Panel
    register : http://127.0.0.1:5000/register_admin
    Use the admin-specific URL provided after running the app to register as admin 
    Or use registered admin credentials on the main login page
    
    
    Admin Features
    
    User management
    System monitoring
    Administrative controls
    
    Note: The same login page handles both user and admin authentication
