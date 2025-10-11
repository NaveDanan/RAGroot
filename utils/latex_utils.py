"""
LaTeX Math Utilities for RAG System
Handles LaTeX inline math delimiters like $P$, $\psi$, etc.
Converts LaTeX commands to Unicode symbols for rich text display.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Comprehensive LaTeX to Unicode mapping for rich text display
LATEX_TO_UNICODE = {
    # Greek letters (lowercase)
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
    r'\theta': 'θ', r'\vartheta': 'ϑ', r'\iota': 'ι', r'\kappa': 'κ',
    r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ',
    r'\pi': 'π', r'\varpi': 'ϖ', r'\rho': 'ρ', r'\varrho': 'ϱ',
    r'\sigma': 'σ', r'\varsigma': 'ς', r'\tau': 'τ', r'\upsilon': 'υ',
    r'\phi': 'φ', r'\varphi': 'φ', r'\chi': 'χ', r'\psi': 'ψ',
    r'\omega': 'ω',
    
    # Greek letters (uppercase)
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
    r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
    
    # Mathematical operators
    r'\times': '×', r'\div': '÷', r'\pm': '±', r'\mp': '∓',
    r'\cdot': '·', r'\ast': '∗', r'\star': '⋆', r'\circ': '∘',
    r'\bullet': '•', r'\oplus': '⊕', r'\ominus': '⊖', r'\otimes': '⊗',
    r'\odot': '⊙', r'\oslash': '⊘', r'\cup': '∪', r'\cap': '∩',
    r'\sqcup': '⊔', r'\sqcap': '⊓', r'\vee': '∨', r'\wedge': '∧',
    
    # Relations
    r'\leq': '≤', r'\le': '≤', r'\geq': '≥', r'\ge': '≥',
    r'\neq': '≠', r'\ne': '≠', r'\approx': '≈', r'\equiv': '≡',
    r'\sim': '∼', r'\simeq': '≃', r'\cong': '≅', r'\propto': '∝',
    r'\ll': '≪', r'\gg': '≫', r'\prec': '≺', r'\succ': '≻',
    r'\preceq': '⪯', r'\succeq': '⪰',
    
    # Set relations
    r'\subset': '⊂', r'\supset': '⊃', r'\subseteq': '⊆', r'\supseteq': '⊇',
    r'\in': '∈', r'\notin': '∉', r'\ni': '∋', r'\notni': '∌',
    r'\emptyset': '∅', r'\varnothing': '∅',
    
    # Logic
    r'\forall': '∀', r'\exists': '∃', r'\nexists': '∄',
    r'\neg': '¬', r'\lnot': '¬', r'\land': '∧', r'\lor': '∨',
    r'\implies': '⟹', r'\impliedby': '⟸', r'\iff': '⟺',
    
    # Arrows
    r'\rightarrow': '→', r'\to': '→', r'\leftarrow': '←', r'\gets': '←',
    r'\leftrightarrow': '↔', r'\Rightarrow': '⇒', r'\Leftarrow': '⇐',
    r'\Leftrightarrow': '⇔', r'\uparrow': '↑', r'\downarrow': '↓',
    r'\updownarrow': '↕', r'\Uparrow': '⇑', r'\Downarrow': '⇓',
    r'\Updownarrow': '⇕', r'\mapsto': '↦', r'\longmapsto': '⟼',
    r'\longrightarrow': '⟶', r'\longleftarrow': '⟵',
    r'\longleftrightarrow': '⟷',
    
    # Calculus & Analysis
    r'\nabla': '∇', r'\partial': '∂', r'\infty': '∞',
    r'\sum': '∑', r'\prod': '∏', r'\coprod': '∐',
    r'\int': '∫', r'\iint': '∬', r'\iiint': '∭', r'\oint': '∮',
    r'\lim': 'lim', r'\sup': 'sup', r'\inf': 'inf',
    r'\limsup': 'lim sup', r'\liminf': 'lim inf',
    
    # Other symbols
    r'\ell': 'ℓ', r'\hbar': 'ℏ', r'\imath': 'ı', r'\jmath': 'ȷ',
    r'\wp': '℘', r'\Re': 'ℜ', r'\Im': 'ℑ', r'\aleph': 'ℵ',
    r'\beth': 'ℶ', r'\gimel': 'ℷ', r'\daleth': 'ℸ',
    
    # Brackets (handled separately but included for completeness)
    r'\langle': '⟨', r'\rangle': '⟩', r'\lfloor': '⌊', r'\rfloor': '⌋',
    r'\lceil': '⌈', r'\rceil': '⌉', r'\llbracket': '⟦', r'\rrbracket': '⟧',
    
    # Misc
    r'\dots': '…', r'\ldots': '…', r'\cdots': '⋯', r'\vdots': '⋮',
    r'\ddots': '⋱', r'\angle': '∠', r'\perp': '⊥', r'\parallel': '∥',
    r'\top': '⊤', r'\bot': '⊥', r'\triangle': '△', r'\square': '□',
    r'\checkmark': '✓', r'\dagger': '†', r'\ddagger': '‡',
}

# Text equivalents for embedding/search (when Unicode isn't ideal)
LATEX_TO_TEXT = {
    r'\alpha': 'alpha', r'\beta': 'beta', r'\gamma': 'gamma', r'\delta': 'delta',
    r'\epsilon': 'epsilon', r'\varepsilon': 'epsilon', r'\zeta': 'zeta', 
    r'\eta': 'eta', r'\theta': 'theta', r'\lambda': 'lambda', r'\mu': 'mu',
    r'\nu': 'nu', r'\xi': 'xi', r'\pi': 'pi', r'\rho': 'rho',
    r'\sigma': 'sigma', r'\tau': 'tau', r'\phi': 'phi', r'\chi': 'chi',
    r'\psi': 'psi', r'\omega': 'omega',
    r'\Gamma': 'Gamma', r'\Delta': 'Delta', r'\Theta': 'Theta',
    r'\Lambda': 'Lambda', r'\Xi': 'Xi', r'\Pi': 'Pi', r'\Sigma': 'Sigma',
    r'\Phi': 'Phi', r'\Psi': 'Psi', r'\Omega': 'Omega',
    r'\infty': 'infinity', r'\times': 'times', r'\div': 'divided by',
    r'\pm': 'plus-minus', r'\leq': 'less than or equal to',
    r'\geq': 'greater than or equal to', r'\neq': 'not equal to',
    r'\approx': 'approximately', r'\equiv': 'equivalent to',
    r'\subset': 'subset of', r'\supset': 'superset of',
    r'\in': 'in', r'\notin': 'not in', r'\forall': 'for all',
    r'\exists': 'exists', r'\nabla': 'nabla', r'\partial': 'partial',
    r'\sum': 'sum', r'\prod': 'product', r'\int': 'integral',
    r'\rightarrow': 'arrow', r'\to': 'to', r'\Rightarrow': 'implies',
    r'\ell': 'ell', r'\log': 'log', r'\ln': 'ln', r'\exp': 'exp',
}

class LaTeXMathHandler:
    """Handle LaTeX math expressions in text."""
    
    def __init__(self, preserve_structure: bool = True, use_unicode: bool = False):
        """
        Initialize LaTeX math handler.
        
        Args:
            preserve_structure: If True, keep mathematical structure visible.
                              If False, replace with simple text equivalents.
            use_unicode: If True, convert LaTeX to Unicode symbols (for rich text display).
                        If False, use text equivalents (for embeddings/search).
        """
        self.preserve_structure = preserve_structure
        self.use_unicode = use_unicode
        
        # Choose appropriate mapping
        self.symbol_map = LATEX_TO_UNICODE if use_unicode else LATEX_TO_TEXT
        
        # Compile regex patterns for efficiency
        self.inline_math_pattern = re.compile(r'\$([^\$]+?)\$')
        self.display_math_pattern = re.compile(r'\$\$([^\$]+?)\$\$')
        
        # Pattern for text formatting commands
        self.textbf_pattern = re.compile(r'\\textbf\{([^}]+)\}')
        self.textit_pattern = re.compile(r'\\textit\{([^}]+)\}')
        self.texttt_pattern = re.compile(r'\\texttt\{([^}]+)\}')
        self.emph_pattern = re.compile(r'\\emph\{([^}]+)\}')
        self.textsc_pattern = re.compile(r'\\textsc\{([^}]+)\}')
        
        # Pattern for URLs
        self.href_pattern = re.compile(r'\\href\{([^}]+)\}\{([^}]+)\}')
        self.url_pattern = re.compile(r'\\url\{([^}]+)\}')
        
    def process_text(self, text: str) -> str:
        """
        Process text to handle LaTeX math delimiters and formatting.
        
        Args:
            text: Input text with LaTeX math and formatting
            
        Returns:
            Processed text with LaTeX converted appropriately
        """
        if not text:
            return text
        
        # Handle text formatting commands first
        text = self._process_text_formatting(text)
        
        # Handle URLs and hyperlinks
        text = self._process_urls(text)
        
        # First handle display math ($$...$$) to avoid conflicts
        text = self.display_math_pattern.sub(
            lambda m: self._process_math_expression(m.group(1), is_display=True),
            text
        )
        
        # Then handle inline math ($...$)
        text = self.inline_math_pattern.sub(
            lambda m: self._process_math_expression(m.group(1), is_display=False),
            text
        )
        
        return text
    
    def _process_text_formatting(self, text: str) -> str:
        """Process LaTeX text formatting commands."""
        # Handle common text formatting
        if self.use_unicode:
            # For rich text, we can use basic formatting hints
            text = self.textbf_pattern.sub(r'**\1**', text)  # Bold
            text = self.textit_pattern.sub(r'*\1*', text)    # Italic
            text = self.emph_pattern.sub(r'*\1*', text)      # Emphasis
            text = self.texttt_pattern.sub(r'`\1`', text)    # Code/monospace
            text = self.textsc_pattern.sub(r'\1', text)      # Small caps (no special formatting)
        else:
            # For embeddings, just extract the content
            text = self.textbf_pattern.sub(r'\1', text)
            text = self.textit_pattern.sub(r'\1', text)
            text = self.emph_pattern.sub(r'\1', text)
            text = self.texttt_pattern.sub(r'\1', text)
            text = self.textsc_pattern.sub(r'\1', text)
        
        return text
    
    def _process_urls(self, text: str) -> str:
        """Process LaTeX URL commands."""
        # \href{url}{text} -> text (url) or just text
        if self.use_unicode:
            text = self.href_pattern.sub(r'\2 (\1)', text)
        else:
            text = self.href_pattern.sub(r'\2', text)
        
        # \url{url} -> url
        text = self.url_pattern.sub(r'\1', text)
        
        return text
    
    def _process_math_expression(self, expr: str, is_display: bool = False) -> str:
        """
        Process a single math expression.
        
        Args:
            expr: The math expression content (without delimiters)
            is_display: Whether this is display math ($$) or inline ($)
            
        Returns:
            Processed expression
        """
        expr = expr.strip()
        
        if not expr:
            return ""
        
        if self.preserve_structure:
            # Keep the math but make it more searchable
            processed = self._expand_latex_symbols(expr)
            
            # For display math, add context
            if is_display:
                return f" [equation: {processed}] "
            else:
                # For inline math, keep it more natural
                return f" {processed} "
        else:
            # Simple replacement - just extract the content
            return f" {expr} "
    
    def _expand_latex_symbols(self, expr: str) -> str:
        """
        Expand LaTeX symbols to Unicode or text equivalents while preserving structure.
        
        Args:
            expr: Math expression
            
        Returns:
            Expression with expanded symbols
        """
        result = expr
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_symbols = sorted(self.symbol_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        # Replace known LaTeX symbols
        for latex_symbol, equivalent in sorted_symbols:
            # Use word boundaries to avoid partial matches
            pattern = re.escape(latex_symbol) + r'(?![a-zA-Z])'
            result = re.sub(pattern, equivalent, result)
        
        # Handle common patterns
        result = self._handle_common_patterns(result)
        
        return result
    
    def _handle_common_patterns(self, expr: str) -> str:
        """Handle common LaTeX patterns."""
        
        if self.use_unicode:
            # For Unicode output, preserve more structure with proper symbols
            # Superscripts: x^2 -> x²
            superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                              '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                              'n': 'ⁿ', 'i': 'ⁱ'}
            
            # Handle simple numeric superscripts
            for digit, superscript in superscript_map.items():
                expr = re.sub(r'\^' + digit + r'(?![0-9])', superscript, expr)
            
            # Subscripts: x_i -> xᵢ
            subscript_map = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                           '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
                           'a': 'ₐ', 'e': 'ₑ', 'i': 'ᵢ', 'o': 'ₒ', 'x': 'ₓ'}
            
            for char, subscript in subscript_map.items():
                expr = re.sub(r'_' + char + r'(?![a-z0-9])', subscript, expr)
            
            # Fractions: \frac{a}{b} -> a/b
            expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', expr)
            
            # Square roots: \sqrt{x} -> √x
            expr = re.sub(r'\\sqrt\{([^}]+)\}', r'√\1', expr)
            
            # Math functions
            expr = re.sub(r'\\log\b', 'log', expr)
            expr = re.sub(r'\\ln\b', 'ln', expr)
            expr = re.sub(r'\\exp\b', 'exp', expr)
            expr = re.sub(r'\\sin\b', 'sin', expr)
            expr = re.sub(r'\\cos\b', 'cos', expr)
            expr = re.sub(r'\\tan\b', 'tan', expr)
            expr = re.sub(r'\\max\b', 'max', expr)
            expr = re.sub(r'\\min\b', 'min', expr)
            
        else:
            # For text output, use descriptive text
            # Superscripts: x^2 -> x squared
            expr = re.sub(r'(\w+)\^2\b', r'\1 squared', expr)
            expr = re.sub(r'(\w+)\^3\b', r'\1 cubed', expr)
            expr = re.sub(r'(\w+)\^\{([^}]+)\}', r'\1 to the power of \2', expr)
            expr = re.sub(r'(\w+)\^(\w+)', r'\1 to the power of \2', expr)
            
            # Subscripts: x_i -> x sub i
            expr = re.sub(r'(\w+)_\{([^}]+)\}', r'\1 sub \2', expr)
            expr = re.sub(r'(\w+)_(\w+)', r'\1 sub \2', expr)
            
            # Fractions: \frac{a}{b} -> a over b
            expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 over \2', expr)
            
            # Square roots: \sqrt{x} -> square root of x
            expr = re.sub(r'\\sqrt\{([^}]+)\}', r'square root of \1', expr)
            
            # Math functions
            expr = re.sub(r'\\log\b', 'log', expr)
            expr = re.sub(r'\\ln\b', 'ln', expr)
            expr = re.sub(r'\\exp\b', 'exp', expr)
        
        # Remove remaining curly braces
        expr = expr.replace('{', '').replace('}', '')
        
        # Handle left/right delimiters
        expr = re.sub(r'\\left\s*', '', expr)
        expr = re.sub(r'\\right\s*', '', expr)
        
        # Remove backslashes from unhandled commands (but preserve the command name)
        expr = re.sub(r'\\([a-zA-Z]+)', r'\1', expr)
        
        return expr
    
    def extract_math_variables(self, text: str) -> List[str]:
        """
        Extract mathematical variables from text.
        Useful for identifying key mathematical concepts.
        
        Args:
            text: Input text with LaTeX math
            
        Returns:
            List of mathematical variables found
        """
        variables = set()
        
        # Find all inline math expressions
        for match in self.inline_math_pattern.finditer(text):
            expr = match.group(1).strip()
            
            # Extract single capital letters (like P, N, etc.)
            variables.update(re.findall(r'\b[A-Z]\b', expr))
            
            # Extract single lowercase letters (common variables)
            variables.update(re.findall(r'\b[a-z]\b', expr))
            
            # Extract Greek letters
            for latex_symbol in self.symbol_map.keys():
                if latex_symbol in expr:
                    variables.add(latex_symbol.replace('\\', ''))
        
        return sorted(list(variables))


def preprocess_for_embedding(text: str) -> str:
    """
    Preprocess text for embedding generation.
    Optimized for semantic search and retrieval.
    Uses text equivalents instead of Unicode for better embedding quality.
    
    Args:
        text: Raw text with LaTeX math
        
    Returns:
        Processed text suitable for embedding
    """
    handler = LaTeXMathHandler(preserve_structure=True, use_unicode=False)
    return handler.process_text(text)


def preprocess_for_display(text: str) -> str:
    """
    Preprocess text for display to users.
    Converts LaTeX to Unicode symbols for rich text display.
    
    Args:
        text: Raw text with LaTeX math and formatting
        
    Returns:
        Processed text with Unicode symbols suitable for display
    """
    handler = LaTeXMathHandler(preserve_structure=True, use_unicode=True)
    return handler.process_text(text)


def latex_to_unicode(text: str) -> str:
    """
    Convert LaTeX commands to Unicode symbols.
    Convenience function for quick conversion.
    
    Args:
        text: Text with LaTeX commands
        
    Returns:
        Text with Unicode symbols
    """
    result = text
    # Sort by length (longest first) to avoid partial replacements
    sorted_symbols = sorted(LATEX_TO_UNICODE.items(), key=lambda x: len(x[0]), reverse=True)
    
    for latex_cmd, unicode_char in sorted_symbols:
        pattern = re.escape(latex_cmd) + r'(?![a-zA-Z])'
        result = re.sub(pattern, unicode_char, result)
    
    return result


if __name__ == "__main__":
    # Test the LaTeX handler with various examples from the arxiv dataset
    test_cases = [
        "Given a planning problem $P$ and formula $\\psi$",
        "We optimize $\\alpha$ and $\\beta$ parameters",
        "The complexity is $O(n^2)$ for this algorithm",
        "Probability distribution $p(x|\\theta)$ is computed",
        "The equation $$\\sum_{i=1}^{n} x_i^2$$ represents variance",
        "Using $\\nabla f(x)$ for gradient descent",
        "Matrix $\\Sigma$ captures covariance structure",
        "We propose $\\lambda$-GRPO for improved training",
        "Model achieves $O(n^{-2r/(2r+1)})$ convergence rate",
        "Using \\textbf{VARL} (\\textbf{V}LM as \\textbf{A}ction advisor)",
        "The limit $N \\to \\infty$ yields Best-of-$\\infty$",
        "Relations: $x \\leq y \\geq z$ and $a \\in S$",
        "Arrows: $f: X \\rightarrow Y$ and $A \\Rightarrow B$",
        "Set operations: $A \\cup B \\cap C \\subseteq D$",
        "Logic: $\\forall x \\exists y$ such that $x \\neq y$",
        "Visit our project at \\href{https://github.com/example}{GitHub}",
        "The function $\\log(x) \\times \\exp(y)$ is smooth",
        "Fractions: $\\frac{a}{b}$ and roots: $\\sqrt{x}$",
        "Greek letters: $\\alpha, \\beta, \\gamma, \\Delta, \\Omega$"
    ]
    
    print("LaTeX Math Handler Test Results")
    print("=" * 100)
    print()
    
    # Test with Unicode output (for display)
    print("=" * 100)
    print("UNICODE OUTPUT MODE (for rich text display)")
    print("=" * 100)
    handler_unicode = LaTeXMathHandler(preserve_structure=True, use_unicode=True)
    
    for i, test_text in enumerate(test_cases, 1):
        processed = handler_unicode.process_text(test_text)
        variables = handler_unicode.extract_math_variables(test_text)
        
        print(f"\n[Test {i}]")
        print(f"Original:  {test_text}")
        print(f"Unicode:   {processed}")
        if variables:
            print(f"Variables: {', '.join(variables)}")
        print("-" * 100)
    
    # Test with text output (for embeddings)
    print("\n\n")
    print("=" * 100)
    print("TEXT OUTPUT MODE (for embeddings and search)")
    print("=" * 100)
    handler_text = LaTeXMathHandler(preserve_structure=True, use_unicode=False)
    
    for i, test_text in enumerate(test_cases[:5], 1):  # Show first 5 examples
        processed = handler_text.process_text(test_text)
        
        print(f"\n[Test {i}]")
        print(f"Original: {test_text}")
        print(f"Text:     {processed}")
        print("-" * 100)
    
    # Test the convenience functions
    print("\n\n")
    print("=" * 100)
    print("CONVENIENCE FUNCTIONS TEST")
    print("=" * 100)
    
    sample = "The model uses $\\alpha \\times \\beta$ with $\\lambda$-GRPO and $N \\to \\infty$"
    print(f"\nOriginal:     {sample}")
    print(f"For Display:  {preprocess_for_display(sample)}")
    print(f"For Embedding: {preprocess_for_embedding(sample)}")
    print(f"Direct Unicode: {latex_to_unicode(sample)}")
