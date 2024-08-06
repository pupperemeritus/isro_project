custom_css_str = """
        <style>
            /* Main container styling */
            .main, .block-container {
            padding: 1rem; /* Reduced padding for a more compact feel */
            border-radius: 8px; /* Slightly less pronounced corners */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Slightly reduced shadow */
            max-width: 100%; /* Ensure responsiveness */
            margin: 0 auto; /* Center align container */
            font-family: sans-serif; /* Modern font for consistency */
        }


            /* Responsive design for smaller screens */
            @media (max-width: 768px) {
                .main {
                    padding: 0.5rem; /* Reduced padding on smaller screens */
                }
            }

            .stSelectbox, .stSlider {
                border-radius: 6px; /* Consistent rounded corners */
                padding: 0.5rem; /* Reduced padding for better fit */
                margin-bottom: 1rem; /* Reduced spacing between elements */
                border: 1px solid #e0e0e0; /* Subtle border */
                transition: border-color 0.3s ease, box-shadow 0.3s ease; /* Smooth transition for border and shadow */
            }


            /* Hover effects */
            .stSelectbox:hover, .stSlider:hover {
                border-color: #bdbdbd; /* Slightly darker border */
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow effect */
            }

            /* Focus states */
            .stSelectbox:focus, .stSlider:focus {
                border-color: #007bff; /* Highlight border color */
                outline: none; /* Remove default outline */
                box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.2); /* Focus shadow for accessibility */
            }

            /* Make sure elements are responsive */
            .stSelectbox select, .stSlider input {
                width: 100%; /* Full width for input elements */
            }

            /* Additional styling for sliders */
            .stSlider input[type="range"] {
                -webkit-appearance: none; /* Remove default appearance */
                width: 100%; /* Full width */
                height: 6px; /* Reduced height of the slider */
                background: transparent; /* No background */
                border-radius: 3px; /* Rounded corners for the track */
                cursor: pointer; /* Pointer cursor */
                outline: none; /* Remove default outline */
                transition: background 0.3s ease; /* Smooth transition for background changes */
            }

            .stSlider input[type="range"]::-webkit-slider-thumb,
            .stSlider input[type="range"]::-moz-range-thumb {
                width: 14px;
                height: 14px;
                border-radius: 50%;
                background: #007bff;
                cursor: pointer;
                transition: background 0.3s ease;
            }

            /* Additional styling for select boxes */
            .stSelectbox select {
                border: 1px solid #e0e0e0; /* Border color */
                padding: 0.5rem; /* Padding inside select */
                border-radius: 6px; /* Rounded corners */
                font-size: 1rem; /* Consistent font size */
                cursor: pointer; /* Pointer cursor */
                transition: border-color 0.3s ease; /* Smooth transition for border color */
            }
 
           .stContainer {
                height: 800px;
                overflow: hidden;
                max-height=100%;
            }
        </style>
    """
