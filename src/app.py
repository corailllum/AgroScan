from streamlit_app import render_app


if __name__ == "__main__":
    render_app("models/resnet50_plantvillage.pth", "AgroScan AI", "#22c55e")

