if not pdf_files and not args.cli:
        console.print("[bold red]Error: No PDF files specified. Use --dir or --files[/bold red]")
        return
    
    # Process documents if specified
    if pdf_files:
        console.print(f"Processing {len(pdf_files)} PDF files")
        analyzer.process_documents(pdf_files)
    else:
        # Try to load documents from output directory
        text_files = glob.glob(os.path.join(args.output, "text", "*.txt"))
        doc_ids = [os.path.basename(f).replace(".txt", "") for f in text_files 
                  if not f.endswith("_pages.json")]
        
        if doc_ids:
            console.print(f"Loading {len(doc_ids)} previously processed documents")
            for doc_id in doc_ids:
                pdf_path = ""  # We don't know the original path
                analyzer.documents[doc_id] = analyzer.process_document(pdf_path)
    
    # Load entity database if exists
    analyzer.load_entity_database()
    
    # Build entity database if needed
    if analyzer.documents and (not analyzer.entities or len(analyzer.entities) == 0):
        analyzer.build_entity_database()
    
    # Enrich entities if requested
    if args.enrich_entities:
        analyzer.enrich_entities()
    
    # Build entity graph if requested
    if args.build_graph:
        analyzer.build_entity_graph()
    
    # Load entity graph if it exists
    graph_path = os.path.join(args.output, "graphs", "entity_graph.gpickle")
    if os.path.exists(graph_path):
        analyzer.entity_graph = nx.read_gpickle(graph_path)
        console.print(f"Loaded entity graph with {analyzer.entity_graph.number_of_nodes()} nodes")
    
    # Start interactive CLI if requested
    if args.cli:
        interactive_cli(analyzer)

if __name__ == "__main__":
    main()
