# FaceCue ML - Complete Pipeline Runner
# Executes all steps of the FaceCue ML pipeline

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

class FaceCuePipeline:
    def __init__(self):
        self.steps_completed = []
        self.data_dir = Path("data")
        self.scripts_dir = Path("scripts")
        
    def run_step(self, step_name, script_path, description):
        """Run a single step of the pipeline"""
        print(f"\n{'='*60}")
        print(f"🚀 {step_name}")
        print(f"{'='*60}")
        print(f"Description: {description}")
        print(f"Script: {script_path}")
        
        try:
            # Check if script exists
            if not script_path.exists():
                print(f"❌ Script not found: {script_path}")
                return False
            
            # Run the script
            result = subprocess.run([sys.executable, str(script_path)], 
                                 capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                print(f"✅ {step_name} completed successfully!")
                print("Output:")
                print(result.stdout)
                self.steps_completed.append(step_name)
                return True
            else:
                print(f"❌ {step_name} failed!")
                print("Error:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error running {step_name}: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete FaceCue ML pipeline"""
        print("🏥 FaceCue ML - Complete Pipeline Execution")
        print("="*60)
        
        # Define pipeline steps
        pipeline_steps = [
            {
                'name': 'Step 3: Data Preprocessing',
                'script': self.scripts_dir / 'step3_preprocess_data.py',
                'description': 'Preprocess lifestyle data, normalize features, encode categorical variables'
            },
            {
                'name': 'Step 4: Model Building',
                'script': self.scripts_dir / 'step4_build_models.py',
                'description': 'Train multiple models (Random Forest, XGBoost, SVM), compare performance'
            },
            {
                'name': 'Step 6: Recommendations System',
                'script': self.scripts_dir / 'step6_recommendations.py',
                'description': 'Generate personalized health recommendations and reports'
            }
        ]
        
        # Run each step
        success_count = 0
        for step in pipeline_steps:
            success = self.run_step(step['name'], step['script'], step['description'])
            if success:
                success_count += 1
            else:
                print(f"\n⚠️ Pipeline stopped at {step['name']}")
                break
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Steps completed: {success_count}/{len(pipeline_steps)}")
        print(f"Completed steps: {', '.join(self.steps_completed)}")
        
        if success_count == len(pipeline_steps):
            print(f"\n🎉 COMPLETE PIPELINE SUCCESS!")
            print(f"✅ All steps completed successfully")
            print(f"✅ FaceCue ML system is ready")
            print(f"✅ Ready for UI deployment (Step 7)")
            
            # Show next steps
            self.show_next_steps()
        else:
            print(f"\n⚠️ PIPELINE INCOMPLETE")
            print(f"Please fix the issues and run again")
        
        return success_count == len(pipeline_steps)
    
    def show_next_steps(self):
        """Show next steps after successful pipeline completion"""
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Install Streamlit: pip install streamlit")
        print(f"2. Run UI: streamlit run scripts/step7_streamlit_ui.py")
        print(f"3. Open browser: http://localhost:8501")
        print(f"4. Test the health prediction system")
        
        print(f"\n📁 GENERATED FILES:")
        generated_files = [
            "data/preprocessed_lifestyle_data.csv",
            "data/best_model.pkl", 
            "data/model_comparison.png",
            "data/health_report_*.json"
        ]
        
        for file_pattern in generated_files:
            if "*" in file_pattern:
                # Check for files matching pattern
                import glob
                matches = glob.glob(file_pattern)
                if matches:
                    print(f"  ✅ {file_pattern} (found {len(matches)} files)")
                else:
                    print(f"  ⏳ {file_pattern} (will be generated)")
            else:
                if Path(file_pattern).exists():
                    print(f"  ✅ {file_pattern}")
                else:
                    print(f"  ⏳ {file_pattern}")
    
    def run_ui_only(self):
        """Run only the UI (Step 7)"""
        print("🎨 Running FaceCue ML UI...")
        
        ui_script = self.scripts_dir / 'step7_streamlit_ui.py'
        
        if not ui_script.exists():
            print(f"❌ UI script not found: {ui_script}")
            return False
        
        try:
            # Check if streamlit is installed
            import streamlit
            print("✅ Streamlit is available")
            
            # Run streamlit
            print("🚀 Starting Streamlit UI...")
            print("📱 Open your browser to: http://localhost:8501")
            
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_script)])
            
        except ImportError:
            print("❌ Streamlit not installed")
            print("📦 Install with: pip install streamlit")
            return False
        except Exception as e:
            print(f"❌ Error running UI: {e}")
            return False
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn'
        ]
        
        optional_packages = [
            'xgboost', 'streamlit'
        ]
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package}")
                missing_required.append(package)
        
        for package in optional_packages:
            try:
                __import__(package)
                print(f"  ✅ {package} (optional)")
            except ImportError:
                print(f"  ⚠️ {package} (optional)")
                missing_optional.append(package)
        
        if missing_required:
            print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
            print(f"📦 Install with: pip install {' '.join(missing_required)}")
            return False
        
        if missing_optional:
            print(f"\n⚠️ Missing optional packages: {', '.join(missing_optional)}")
            print(f"📦 Install with: pip install {' '.join(missing_optional)}")
        
        print(f"\n✅ All required dependencies are available!")
        return True

def main():
    """Main function"""
    pipeline = FaceCuePipeline()
    
    print("🏥 FaceCue ML - Pipeline Manager")
    print("="*50)
    
    # Check dependencies first
    if not pipeline.check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Show options
    print(f"\n📋 Available Options:")
    print(f"1. Run Complete Pipeline (Steps 3, 4, 6)")
    print(f"2. Run UI Only (Step 7)")
    print(f"3. Check Dependencies")
    
    try:
        choice = input(f"\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            pipeline.run_complete_pipeline()
        elif choice == "2":
            pipeline.run_ui_only()
        elif choice == "3":
            pipeline.check_dependencies()
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print(f"\n👋 Pipeline execution cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
