"""
GitHub Readiness Checker
Verifies that the project is ready for GitHub upload
"""

import os
from pathlib import Path
from typing import List, Tuple

# ANSI color codes for Windows PowerShell
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class GitHubReadinessChecker:
    """Check if project is ready for GitHub upload"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues = []
        self.warnings = []
        self.passed_checks = []
    
    def check_essential_files(self) -> bool:
        """Check for essential files"""
        essential_files = [
            'README.md',
            'LICENSE',
            '.gitignore',
            'requirements.txt',
            'setup.py',
            'flask_app.py',
            'mlops_pipeline.py'
        ]
        
        all_present = True
        for file in essential_files:
            file_path = self.project_root / file
            if file_path.exists():
                self.passed_checks.append(f"✓ {file} exists")
            else:
                self.issues.append(f"✗ Missing essential file: {file}")
                all_present = False
        
        return all_present
    
    def check_directory_structure(self) -> bool:
        """Check for required directories"""
        required_dirs = [
            'src',
            'src/data',
            'src/features',
            'src/models',
            'src/visualization',
            'src/utils',
            'tests',
            'notebooks',
            'data',
            'models',
            'logs'
        ]
        
        all_present = True
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self.passed_checks.append(f"✓ {dir_name}/ directory exists")
            else:
                self.issues.append(f"✗ Missing directory: {dir_name}")
                all_present = False
        
        return all_present
    
    def check_gitignore(self) -> bool:
        """Check .gitignore content"""
        gitignore_path = self.project_root / '.gitignore'
        
        if not gitignore_path.exists():
            self.issues.append("✗ .gitignore file missing")
            return False
        
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        required_patterns = [
            '__pycache__',
            '*.pyc',
            'venv/',
            '.env',
            '*.pkl',
            '*.log'
        ]
        
        all_present = True
        for pattern in required_patterns:
            if pattern in content:
                self.passed_checks.append(f"✓ .gitignore includes {pattern}")
            else:
                self.warnings.append(f"⚠ .gitignore might be missing: {pattern}")
                all_present = False
        
        return all_present
    
    def check_personal_info(self) -> bool:
        """Check if personal information is updated"""
        readme_path = self.project_root / 'README.md'
        
        if not readme_path.exists():
            self.issues.append("✗ README.md not found")
            return False
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for placeholder text
        placeholders = [
            'your.email@example.com',
            'yourusername',
            'Your Name'
        ]
        
        has_placeholders = False
        for placeholder in placeholders:
            if placeholder in content:
                self.warnings.append(f"⚠ README contains placeholder: {placeholder}")
                has_placeholders = True
        
        # Check for actual information
        if 'Muhammad Farooq' in content and 'mfarooqshafee333@gmail.com' in content:
            self.passed_checks.append("✓ Personal information updated in README")
            return True
        else:
            if not has_placeholders:
                self.warnings.append("⚠ Personal information might not be updated")
            return not has_placeholders
    
    def check_documentation(self) -> bool:
        """Check for documentation files"""
        doc_files = [
            'README.md',
            'GETTING_STARTED.md',
            'PROJECT_SUMMARY.md',
            'CONTRIBUTING.md',
            'ARCHITECTURE.md'
        ]
        
        all_present = True
        for file in doc_files:
            file_path = self.project_root / file
            if file_path.exists():
                self.passed_checks.append(f"✓ {file} documentation exists")
            else:
                self.warnings.append(f"⚠ Optional documentation missing: {file}")
        
        return True  # Documentation is helpful but not critical
    
    def check_tests(self) -> bool:
        """Check for test files"""
        test_dir = self.project_root / 'tests'
        
        if not test_dir.exists():
            self.warnings.append("⚠ tests/ directory not found")
            return False
        
        test_files = list(test_dir.glob('test_*.py'))
        
        if len(test_files) >= 3:
            self.passed_checks.append(f"✓ Found {len(test_files)} test files")
            return True
        else:
            self.warnings.append(f"⚠ Only {len(test_files)} test files found")
            return False
    
    def check_docker(self) -> bool:
        """Check for Docker files"""
        docker_files = ['Dockerfile', 'docker-compose.yml']
        
        all_present = True
        for file in docker_files:
            file_path = self.project_root / file
            if file_path.exists():
                self.passed_checks.append(f"✓ {file} exists")
            else:
                self.warnings.append(f"⚠ {file} not found (optional for deployment)")
                all_present = False
        
        return True  # Docker is optional
    
    def check_github_actions(self) -> bool:
        """Check for CI/CD configuration"""
        actions_dir = self.project_root / '.github' / 'workflows'
        
        if actions_dir.exists() and list(actions_dir.glob('*.yml')):
            self.passed_checks.append("✓ GitHub Actions workflow configured")
            return True
        else:
            self.warnings.append("⚠ GitHub Actions not configured (optional)")
            return True
    
    def estimate_project_size(self):
        """Estimate project size"""
        total_files = 0
        total_lines = 0
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip virtual environment and cache
            dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'node_modules']]
            
            for file in files:
                if file.endswith(('.py', '.md', '.yml', '.yaml', '.txt', '.json')):
                    total_files += 1
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += len(f.readlines())
                    except:
                        pass
        
        self.passed_checks.append(f"✓ Project contains {total_files} files")
        self.passed_checks.append(f"✓ Total lines of code: ~{total_lines}")
    
    def run_all_checks(self):
        """Run all checks"""
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}GitHub Readiness Checker{RESET}")
        print(f"{BLUE}{'='*80}{RESET}\n")
        
        checks = [
            ("Essential Files", self.check_essential_files),
            ("Directory Structure", self.check_directory_structure),
            (".gitignore Configuration", self.check_gitignore),
            ("Personal Information", self.check_personal_info),
            ("Documentation", self.check_documentation),
            ("Test Suite", self.check_tests),
            ("Docker Configuration", self.check_docker),
            ("GitHub Actions", self.check_github_actions)
        ]
        
        print(f"{BLUE}Running checks...{RESET}\n")
        
        for check_name, check_func in checks:
            print(f"Checking {check_name}...")
            check_func()
        
        # Estimate project size
        self.estimate_project_size()
        
        # Print results
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{GREEN}PASSED CHECKS ({len(self.passed_checks)}){RESET}")
        print(f"{BLUE}{'='*80}{RESET}")
        for check in self.passed_checks:
            print(f"{GREEN}{check}{RESET}")
        
        if self.warnings:
            print(f"\n{BLUE}{'='*80}{RESET}")
            print(f"{YELLOW}WARNINGS ({len(self.warnings)}){RESET}")
            print(f"{BLUE}{'='*80}{RESET}")
            for warning in self.warnings:
                print(f"{YELLOW}{warning}{RESET}")
        
        if self.issues:
            print(f"\n{BLUE}{'='*80}{RESET}")
            print(f"{RED}ISSUES ({len(self.issues)}){RESET}")
            print(f"{BLUE}{'='*80}{RESET}")
            for issue in self.issues:
                print(f"{RED}{issue}{RESET}")
        
        # Final verdict
        print(f"\n{BLUE}{'='*80}{RESET}")
        if not self.issues:
            print(f"{GREEN}✓ PROJECT IS READY FOR GITHUB UPLOAD!{RESET}")
            print(f"{BLUE}{'='*80}{RESET}\n")
            print(f"{GREEN}Next steps:{RESET}")
            print(f"1. Run: {YELLOW}git init{RESET}")
            print(f"2. Run: {YELLOW}git add .{RESET}")
            print(f"3. Run: {YELLOW}git commit -m 'Initial commit'{RESET}")
            print(f"4. Create repository at: {YELLOW}https://github.com/Muhammad-Farooq-13{RESET}")
            print(f"5. Run: {YELLOW}git remote add origin <your-repo-url>{RESET}")
            print(f"6. Run: {YELLOW}git push -u origin main{RESET}")
            print(f"\n{GREEN}See UPLOAD_INSTRUCTIONS.md for detailed steps!{RESET}")
        else:
            print(f"{RED}✗ PLEASE FIX ISSUES BEFORE UPLOADING{RESET}")
            print(f"{BLUE}{'='*80}{RESET}")
        
        print()


def main():
    """Main function"""
    checker = GitHubReadinessChecker()
    checker.run_all_checks()


if __name__ == "__main__":
    main()
