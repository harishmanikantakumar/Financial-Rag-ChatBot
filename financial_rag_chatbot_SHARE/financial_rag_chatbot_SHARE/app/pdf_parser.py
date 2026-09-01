import pdfplumber
import re

class PDFParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = ""
        self.financial_data = {}
        self._parse_pdf()
    
    def _parse_pdf(self):
        """Parse PDF and extract text"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        self.text += page_text + "\n"
                
                # Extract specific metrics
                self._extract_revenue()
                self._extract_net_income()
                self._extract_operating_income()
                self._extract_rnd()
                
        except Exception as e:
            print(f"Error parsing PDF: {e}")
    
    def _is_valid_number(self, val):
        """Check if number is a valid financial value (not a year, not too small)"""
        # Skip years (2020-2030)
        if 2020 <= val <= 2030:
            return False
        # Skip numbers that are too small (likely page numbers or years)
        if val < 1000:
            return False
        # Skip numbers that are too large (unlikely for millions)
        if val > 1000000000:
            return False
        return True
    
    def _extract_revenue(self):
        """Extract TOTAL revenue - fixed to skip years properly"""
        lines = self.text.split('\n')
        
        # Look for the income statement section
        income_statement_start = -1
        for i, line in enumerate(lines):
            if 'consolidated statements of operations' in line.lower() or 'statements of operations' in line.lower():
                income_statement_start = i
                break
        
        # If found income statement, search there
        if income_statement_start != -1:
            search_lines = lines[income_statement_start:income_statement_start + 30]
        else:
            search_lines = lines
        
        for i, line in enumerate(search_lines):
            line_lower = line.lower()
            
            # Look for total revenue indicators
            revenue_indicators = ['total net sales', 'total revenues', 'total revenue']
            
            # Skip sub-categories
            skip_terms = ['u.s.', 'china', 'product', 'service', 'server', 'automotive', 
                         'leasing', 'regulatory', 'windows', 'office', 'linkedin', 'gaming']
            
            if any(term in line_lower for term in skip_terms):
                continue
            
            for indicator in revenue_indicators:
                if indicator in line_lower:
                    # Get this line and next few lines
                    context = ' '.join(search_lines[i:i+5])
                    
                    # Find all numbers
                    numbers = re.findall(r'[\d,]+', context)
                    
                    valid_numbers = []
                    for num in numbers:
                        try:
                            val = int(num.replace(',', ''))
                            if self._is_valid_number(val):
                                valid_numbers.append(val)
                        except:
                            pass
                    
                    if valid_numbers:
                        # For revenue, take the LARGEST number (should be total)
                        revenue_value = max(valid_numbers)
                        self.financial_data['revenue'] = [revenue_value]
                        print(f"✅ Revenue: ${revenue_value:,} million")
                        return
    
    def _extract_net_income(self):
        """Extract Net Income - fixed"""
        lines = self.text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for net income
            if 'net income' in line_lower and 'per share' not in line_lower:
                context = ' '.join(lines[i:i+5])
                numbers = re.findall(r'[\d,]+', context)
                
                valid_numbers = []
                for num in numbers:
                    try:
                        val = int(num.replace(',', ''))
                        if self._is_valid_number(val):
                            valid_numbers.append(val)
                    except:
                        pass
                
                if valid_numbers:
                    self.financial_data['net_income'] = [valid_numbers[0]]
                    print(f"✅ Net Income: {valid_numbers[0]:,} million")
                    return
    
    def _extract_operating_income(self):
        """Extract Operating Income - fixed"""
        lines = self.text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for operating income (but NOT net income)
            if 'operating income' in line_lower and 'net income' not in line_lower:
                context = ' '.join(lines[i:i+5])
                numbers = re.findall(r'[\d,]+', context)
                
                valid_numbers = []
                for num in numbers:
                    try:
                        val = int(num.replace(',', ''))
                        if self._is_valid_number(val):
                            valid_numbers.append(val)
                    except:
                        pass
                
                if valid_numbers:
                    self.financial_data['operating_income'] = [valid_numbers[0]]
                    print(f"✅ Operating Income: {valid_numbers[0]:,} million")
                    return
    
    def _extract_rnd(self):
        """Extract Research and Development - fixed"""
        lines = self.text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for R&D
            if 'research and development' in line_lower or 'r&d' in line_lower:
                context = ' '.join(lines[i:i+5])
                numbers = re.findall(r'[\d,]+', context)
                
                valid_numbers = []
                for num in numbers:
                    try:
                        val = int(num.replace(',', ''))
                        if self._is_valid_number(val):
                            valid_numbers.append(val)
                    except:
                        pass
                
                if valid_numbers:
                    self.financial_data['research_development'] = [valid_numbers[0]]
                    print(f"✅ R&D: {valid_numbers[0]:,} million")
                    return
    
    def get_value(self, metric, year=None):
        """Get value for a specific metric and year"""
        metric_map = {
            'revenue': 'revenue',
            'net sales': 'revenue',
            'total net sales': 'revenue',
            'net income': 'net_income',
            'operating income': 'operating_income',
            'research and development': 'research_development',
            'r&d': 'research_development',
            'operating expenses': 'operating_expenses',
        }
        
        metric_key = metric_map.get(metric.lower(), metric.lower().replace(' ', '_'))
        
        if metric_key not in self.financial_data:
            return None
        
        values = self.financial_data[metric_key]
        
        if not values:
            return None
        
        if year:
            year_map = {2023: 0, 2022: 1, 2021: 2}
            idx = year_map.get(year, 0)
            if idx < len(values):
                return values[idx]
        
        return values[0] if values else None
    
    def get_all_data(self):
        """Return all extracted financial data"""
        return self.financial_data

def parse_pdf_for_rag(pdf_path):
    """Main function to parse PDF and return data for RAG"""
    parser = PDFParser(pdf_path)
    return parser