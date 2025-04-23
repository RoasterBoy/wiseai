#!/usr/bin/env node
// PDF Page Extractor - Extract pages from a PDF that match a search string

const fs = require('fs');
const path = require('path');
const { program } = require('commander');
const PDFDocument = require('pdf-lib').PDFDocument;
const pdf = require('pdf-parse');

program
  .name('pdf-extractor')
  .description('Extract pages from a PDF that contain a specific string')
  .version('1.0.0')
  .requiredOption('-i, --input <file>', 'Input PDF file path')
  .requiredOption('-s, --search <string>', 'String to search for')
  .option('-o, --output <file>', 'Output PDF file path')
  .option('-c, --case-sensitive', 'Enable case-sensitive matching', false)
  .option('-v, --verbose', 'Enable verbose output', false)
  .option('-r, --regex', 'Use regular expression for search', false)
  .option('--char-codes', 'Show character codes for the search string', false)
  .option('--debug-page <number>', 'Debug text extraction for a specific page', -1)
  .parse(process.argv);

const options = program.opts();

// Attempt to debug specific extraction issues
async function debugPage() {
  console.log(`DEBUG MODE: Analyzing text extraction for page ${options.debugPage}`);
  try {
    const pageOptions = {
      max: 1,
      page: options.debugPage
    };
    
    const data = await pdf(inputPdfBytes, pageOptions);
    const pageText = data.text || '';
    
    console.log("==== START OF PAGE TEXT ====");
    console.log(pageText);
    console.log("==== END OF PAGE TEXT ====");
    
    console.log("\nCharacter codes for the first 100 characters:");
    const charCodes = Array.from(pageText.substring(0, 100)).map(c => `${c}: ${c.charCodeAt(0)}`);
    console.log(charCodes.join('\n'));
    
    if (options.search) {
      console.log(`\nSearching for: "${options.search}"`);
      if (options.charCodes) {
        console.log("Character codes for search string:");
        const searchCharCodes = Array.from(options.search).map(c => `${c}: ${c.charCodeAt(0)}`);
        console.log(searchCharCodes.join('\n'));
      }
      
      const normalizedText = pageText.replace(/\s+/g, ' ').trim();
      
      // Search with various methods
      const methods = [
        { name: "Exact match", result: pageText.includes(options.search) },
        { name: "Case-insensitive", result: pageText.toLowerCase().includes(options.search.toLowerCase()) },
        { name: "Normalized text", result: normalizedText.includes(options.search) },
        { name: "Normalized & case-insensitive", result: normalizedText.toLowerCase().includes(options.search.toLowerCase()) }
      ];
      
      console.log("\nSearch results:");
      methods.forEach(method => {
        console.log(`- ${method.name}: ${method.result ? "FOUND" : "Not found"}`);
      });
      
      // Look for partials
      console.log("\nSearching for partial matches:");
      for (let i = 2; i <= options.search.length; i++) {
        const partial = options.search.substring(0, i);
        const found = pageText.indexOf(partial);
        if (found >= 0) {
          console.log(`- Found "${partial}" at position ${found}`);
          const context = pageText.substring(Math.max(0, found - 20), Math.min(pageText.length, found + partial.length + 20));
          console.log(`  Context: "${context}"`);
        } else {
          console.log(`- "${partial}" not found`);
        }
      }
    }
  } catch (error) {
    console.error(`Error debugging page ${options.debugPage}:`, error.message);
    process.exit(1);
  }
}

async function extractPagesWithString() {
  try {
    // Validate input file
    if (!fs.existsSync(options.input)) {
      console.error(`Error: Input file '${options.input}' does not exist.`);
      process.exit(1);
    }

    // Default output filename if not provided
    if (!options.output) {
      const inputPath = path.parse(options.input);
      options.output = path.join(inputPath.dir, `${inputPath.name}_extracted${inputPath.ext}`);
    }

    // Read the input PDF file
    const inputPdfBytes = fs.readFileSync(options.input);
    
    // Debug specific page if requested
    if (options.debugPage > 0) {
      await debugPage();
      process.exit(0);
    }
    
    // Try to extract text from the entire document first
    if (options.verbose) console.log("Extracting text from the entire document...");
    const fullPdfData = await pdf(inputPdfBytes);
    const fullText = fullPdfData.text;
    
    // Check if search string exists in the full document
    const searchStringFull = options.caseSensitive ? options.search : options.search.toLowerCase();
    const fullTextForSearch = options.caseSensitive ? fullText : fullText.toLowerCase();
    
    const fullDocumentHasMatch = options.regex 
      ? new RegExp(options.search, options.caseSensitive ? '' : 'i').test(fullTextForSearch)
      : fullTextForSearch.includes(searchStringFull);
    
    if (!fullDocumentHasMatch) {
      console.log(`Warning: The search string "${options.search}" was not found in the full document text.`);
      console.log("This could indicate extraction issues. Proceeding with page-by-page search anyway...");
      
      // Try debugging some information about the text
      if (options.verbose) {
        console.log("First 100 characters of extracted text:");
        console.log(fullText.substring(0, 100));
        console.log("Character codes of the first 20 characters:");
        console.log(Array.from(fullText.substring(0, 20)).map(c => c.charCodeAt(0)));
        
        // Try alternative forms of the search string
        const searchChars = Array.from(options.search);
        console.log(`Character codes of search string "${options.search}":`);
        console.log(searchChars.map(c => c.charCodeAt(0)));
      }
    } else if (options.verbose) {
      console.log(`Good news! Found "${options.search}" in the full document. Proceeding to locate specific pages...`);
    }
    
    // Load the PDF for page copying
    const pdfDoc = await PDFDocument.load(inputPdfBytes);
    const pageCount = pdfDoc.getPageCount();
    
    // Create a new PDF document for the output
    const outputPdfDoc = await PDFDocument.create();
    
    // Search each page for the string
    const matchingPages = [];
    
    if (options.verbose) console.log(`PDF has ${pageCount} pages. Searching each page...`);
    
    // Process each page individually
    for (let i = 0; i < pageCount; i++) {
      if (options.verbose) console.log(`Processing page ${i + 1}/${pageCount}...`);
      
      try {
        // Use multiple extraction strategies
        let pageText = "";
        let isMatch = false;
        
        // Strategy 1: Default extraction
        try {
          const pageOptions = {
            max: 1,
            page: i + 1
          };
          
          const data = await pdf(inputPdfBytes, pageOptions);
          pageText = data.text || '';
          
          // Try both original and normalized text
          const originalPageText = pageText;
          // Also try with normalized spaces
          pageText = pageText.replace(/\s+/g, ' ').trim();
          
          if (options.regex) {
            // Use regular expression search
            try {
              const flags = options.caseSensitive ? '' : 'i';
              const regex = new RegExp(options.search, flags);
              isMatch = regex.test(originalPageText) || regex.test(pageText);
            } catch (e) {
              console.error(`Invalid regular expression: ${options.search}`);
              process.exit(1);
            }
          } else {
            // Use string search
            const searchString = options.caseSensitive ? options.search : options.search.toLowerCase();
            const originalTextForSearch = options.caseSensitive ? originalPageText : originalPageText.toLowerCase();
            const normalizedTextForSearch = options.caseSensitive ? pageText : pageText.toLowerCase();
            
            isMatch = originalTextForSearch.includes(searchString) || 
                     normalizedTextForSearch.includes(searchString);
          }
        } catch (e) {
          if (options.verbose) console.log(`Error in extraction strategy 1: ${e.message}`);
        }
        
        // Strategy 2: Try with enhanced rendering if first strategy didn't match
        if (!isMatch) {
          try {
            const enhancedOptions = {
              max: 1,
              page: i + 1,
              pagerender: function(pageData) {
                const viewport = pageData.getViewport({ scale: 1.5 });
                const canvasFactory = {
                  create: function(width, height) {
                    return { width, height };
                  },
                  reset: function(canvasAndContext, width, height) {
                    canvasAndContext.width = width;
                    canvasAndContext.height = height;
                  },
                  destroy: function(canvasAndContext) {}
                };
                const renderContext = {
                  canvasContext: {},
                  viewport: viewport,
                  canvasFactory: canvasFactory
                };
                return pageData.render(renderContext).promise;
              }
            };
            
            const enhancedData = await pdf(inputPdfBytes, enhancedOptions);
            const enhancedText = enhancedData.text || '';
            
            if (options.regex) {
              const flags = options.caseSensitive ? '' : 'i';
              const regex = new RegExp(options.search, flags);
              isMatch = regex.test(enhancedText);
            } else {
              const searchString = options.caseSensitive ? options.search : options.search.toLowerCase();
              const textForSearch = options.caseSensitive ? enhancedText : enhancedText.toLowerCase();
              isMatch = textForSearch.includes(searchString);
            }
          } catch (e) {
            if (options.verbose) console.log(`Error in extraction strategy 2: ${e.message}`);
          }
        }
        
        // Strategy 3: Check if this page's content exists in the portion of the full document text
        // This helps when individual page extraction fails but full document extraction works
        if (!isMatch && fullDocumentHasMatch) {
          try {
            // Estimate this page's text in the full document
            // This is not exact but can help with simple PDFs
            const avgCharsPerPage = fullText.length / pageCount;
            const startPos = Math.max(0, Math.floor(i * avgCharsPerPage - avgCharsPerPage/2));
            const endPos = Math.min(fullText.length, Math.floor((i+1) * avgCharsPerPage + avgCharsPerPage/2));
            const estimatedPageText = fullText.substring(startPos, endPos);
            
            if (options.regex) {
              const flags = options.caseSensitive ? '' : 'i';
              const regex = new RegExp(options.search, flags);
              isMatch = regex.test(estimatedPageText);
            } else {
              const searchString = options.caseSensitive ? options.search : options.search.toLowerCase();
              const textForSearch = options.caseSensitive ? estimatedPageText : estimatedPageText.toLowerCase();
              isMatch = textForSearch.includes(searchString);
            }
            
            if (isMatch && options.verbose) {
              console.log("Match found using full-document text estimation");
            }
          } catch (e) {
            if (options.verbose) console.log(`Error in extraction strategy 3: ${e.message}`);
          }
        }
        
        if (isMatch) {
          matchingPages.push(i);
          if (options.verbose) console.log(`✓ Match found on page ${i + 1}`);
        } else if (options.verbose) {
          console.log(`✗ No match on page ${i + 1}`);
        }
      } catch (error) {
        console.error(`Error processing page ${i + 1}:`, error.message);
      }
    }
    
    if (matchingPages.length === 0) {
      console.log(`No pages containing "${options.search}" were found.`);
      process.exit(0);
    }
    
    // Copy matching pages to the output document
    if (options.verbose) console.log('Copying matching pages to new document...');
    
    const copiedPages = await outputPdfDoc.copyPages(pdfDoc, matchingPages);
    
    copiedPages.forEach(page => {
      outputPdfDoc.addPage(page);
    });
    
    // Save the output PDF
    const outputPdfBytes = await outputPdfDoc.save();
    fs.writeFileSync(options.output, outputPdfBytes);
    
    console.log(`Extracted ${matchingPages.length} page(s) to ${options.output}`);
    console.log(`Matching pages: ${matchingPages.map(p => p + 1).join(', ')}`);
    
  } catch (error) {
    console.error('Error processing PDF:', error.message);
    process.exit(1);
  }
}

// Execute the main function
extractPagesWithString();