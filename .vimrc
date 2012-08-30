syntax enable
filetype plugin on
filetype indent on
set expandtab
set autoindent
set bs=2

" Disable the arrow keys
map <up> <nop>
map <down> <nop>
map <left> <nop>
map <right> <nop>
imap <up> <nop>
imap <down> <nop>
imap <left> <nop>
imap <right> <nop>

" Highlight search results
set incsearch
set hlsearch

" Some of Derek Wyatt's config (https://github.com/derekwyatt/vim-config/blob/master/vimrc)

" tell VIM to always put a status line in, even if there is only one window
set laststatus=2

" Andrew's ninja-folding
" - zM (collapse all folds)
" - zR (expand all folds)
" - <space> expands or collapses a given fold
set foldmethod=indent foldlevel=999
nmap <silent> <space> :exe 'silent!normal!za' <cr>
highlight Folded ctermbg=Black ctermfg=DarkBlue

" Show line numbers... (nah)
set nu

" Richard's gosu vim hints for astyle
set modeline "enable reading of vim settings from specially formatted comments
let c_space_errors=1 "highlight trailing whitespace
set ts=4 sw=4 expandtab "make tab key conform to coding guidelines
" let w:m1=matchadd('Search', '\%>80v.\%<122v', -1) "highlight lines >80 chars in yellow
let w:m2=matchadd('ErrorMsg', '\%>120v.\+', -1) " highlight lines > 120 chars in red
let w:m3=matchadd('ErrorMsg', '\t', -1) " highlight tabs in red
set cindent "indent as you type
set cinoptions='t0g0' "tweak indenting style
set tw=120

" Easily switch between .h and .cpp in the same directory, or ../src/xyz.cpp <-> ../include/xyz.h
map <F4> :e %:p:s,.h$,.X123X,:s,.cpp$,.h,:s,.X123X$,.cpp,:s,include/\(\w\+\.\w\+\)$,XABCX/\1,:s,src/\(\w\+\.\w\+\)$,include/\1,:s,XABCX,src,<CR>
