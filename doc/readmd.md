Make the docs, from this directory:

```
julia --project=. make.jl
```

If you have LiveServer installed in your base environment, you can view the docs like so:

```
julia --project=. -e 'using LiveServer; serve(; dir = "build", launch_browser = true)'
```
